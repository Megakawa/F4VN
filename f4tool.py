#!/usr/bin/env python3
"""
f4tool.py (v2) - Công cụ việt hóa Fallout 4 string tables bằng model local (Ollama).

Cấu trúc thư mục:
    in/    <- đặt Fallout4_en.STRINGS, .DLSTRINGS, .ILSTRINGS (bản tiếng Anh gốc) vào đây
    work/  <- file trung gian (tự tạo)
    out/   <- file thành phẩm (tự tạo)

Lệnh:
    python f4tool.py selftest
    python f4tool.py extract
    python f4tool.py translate --model qwen3.5:9b [--only S|D|I] [--limit 100] [--workers 2]
    python f4tool.py purge  [--only S]   (xóa khỏi cache các dòng không đạt kiểm tra hiện tại, để dịch lại)
    python f4tool.py build
    python f4tool.py check
Có thể dừng (Ctrl+C) và chạy lại, tiến độ được lưu trong work/cache.jsonl.

Điểm mới ở v2:
  * Model phải chép lại câu gốc (dạng `số|gốc ⇒ dịch`) để script phát hiện bản dịch bị lệch dòng.
  * Chuỗi nhiều dòng (\\r\\n) được tách thành từng dòng để dịch rồi ghép lại.
  * Chỉ giữ nguyên thẻ kỹ thuật và phím thật; nhãn như [Lie], [Optional], [READY] được dịch.
  * Từ chối bản dịch dùng "mày/tao" ở STRINGS/DLSTRINGS, bản dịch giữ nguyên tiếng Anh, độ dài bất thường.
"""
import argparse, json, os, re, struct, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT = {'S': 'STRINGS', 'D': 'DLSTRINGS', 'I': 'ILSTRINGS'}
HAS_LEN = {'S': False, 'D': True, 'I': True}
IN, WORK, OUT = os.path.join(SCRIPT_DIR, 'in'), os.path.join(SCRIPT_DIR, 'work'), os.path.join(SCRIPT_DIR, 'out')
GUIDE_FILE = os.path.join(SCRIPT_DIR, 'guide_local.txt')

# Phím / nút thật trong game (được thay bằng biểu tượng) -> phải giữ nguyên.
KEYS = set("""Click Accept Activate Cancel XButton YButton AButton BButton LeftStick RightStick Leftstick Rightstick
LShoulder RShoulder LTrigger RTrigger Left Right Up Down Forward Back Pipboy PipBoy VATS Melee ReadyWeapon
PrimaryAttack SecondaryAttack RotateLock RotatePick TogglePOV Sneak Jump Sprint pagebreak _Dpad_None Space Mouse2
Pick NextTarget PrevTarget L3 R3 ToggleAlwaysRun NextPerk PrevPerk QCOpenTransferMenu QCTakeItem ApplyCritical
Pause Automove Quickkeys""".split())
KEY_PATTERN = re.compile(r"Quickkey\w*|MultiActivate\w|Strafe\w+")
TAG_RE = re.compile(
    r"</?(?:p|font|i|b|u|br)\b[^>]*>"
    r"|<(?:Alias|alias|Global|global|Token|token)[^>]*>"
    r"|</?(?:mag|Mag|dur)>"
    r"|<[A-Z]\.[A-Za-z]+>"
    r"|<(?:cough|zot|bzzzzt?|sob|sigh|Growl|ERROR|sinister laugh)>"
    r"|\[[A-Za-z0-9_]+\]"
    r"|%[A-Za-z%]+"
    r"|[=*\-_#:]{3,}")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
NL_RE = re.compile(r'(\r\n|\n|\r)')
HOSTILE_RE = re.compile(r"(?<!lông )(?<!chân )(?<!Lông )(?<!Chân )\b(mày|tao)\b", re.I)  # "lông mày" (brow) hợp lệ


# ---------------------------------------------------------------- đọc / ghi
def esc(t):
    return t.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')


def unesc(t):
    out, i = [], 0
    while i < len(t):
        c = t[i]
        if c == '\\' and i + 1 < len(t):
            n = t[i + 1]
            out.append({'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}.get(n, '\\' + n))
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def read_table(path, has_len):
    d = open(path, 'rb').read()
    cnt = struct.unpack('<I', d[:4])[0]
    base = 8 + cnt * 8
    rows = []
    for i in range(cnt):
        sid, off = struct.unpack('<II', d[8 + i * 8:16 + i * 8])
        p = base + off
        if has_len:
            ln = struct.unpack('<I', d[p:p + 4])[0]
            raw = d[p + 4:p + 4 + ln]
        else:
            raw = d[p:d.index(b'\0', p)]
        raw = raw.rstrip(b'\0')
        try:
            txt = raw.decode('utf-8')
        except UnicodeDecodeError:  # một số chuỗi gốc dùng mã Windows-1252
            txt = raw.decode('cp1252', errors='replace')
        rows.append((sid, txt))
    return rows


def write_table(path, rows, has_len):
    cnt = len(rows)
    data, dirs, off = bytearray(), bytearray(), 0
    for sid, text in rows:
        b = text.encode('utf-8') + b'\0'
        dirs += struct.pack('<II', sid, off)
        rec = (struct.pack('<I', len(b)) + b) if has_len else b
        data += rec
        off += len(rec)
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', cnt, len(data)) + dirs + data)


def find_input(key):
    for fn in os.listdir(IN):
        if fn.upper().endswith('.' + EXT[key]):
            return os.path.join(IN, fn)
    return None


# ---------------------------------------------------------------- lọc / tách chuỗi
def skip(t):
    s = t.strip()
    if not s:
        return True
    if re.fullmatch(r"[A-Za-z0-9_:.]+", s) and re.search(r"[a-z][A-Z]|_", s):
        return True  # ID engine
    if not re.search(r"[A-Za-z]", s):
        return True  # chỉ số / ký hiệu
    if s.startswith('***') and s.endswith('***'):
        return True  # ghi chú dev
    if s.startswith('$') and not re.search(r'\s', s):
        return True  # key kiểu $Voice
    if not re.search(r'\s', s) and (re.search(r'[\\/]', s) or re.search(r'\.(dds|nif|wav|xwm|fuz|swf|txt|esp|esm|bsa|hkx)$', s, re.I)):
        return True  # đường dẫn / tên file
    if s.lower() in ('st', 'nd', 'rd', 'th'):
        return True  # hậu tố số thứ tự
    return False


def needs_split(t):
    return bool(NL_RE.search(t))


def unit_skip(core):
    """Dòng riêng lẻ trong chuỗi nhiều dòng có cần bỏ qua không."""
    return skip(core) or '<User' in core  # dòng "<User ... signed in>" là phần khung terminal, giữ nguyên


def units_of(t):
    """Các đơn vị cần dịch của một chuỗi gốc."""
    if not needs_split(t):
        return [] if skip(t) else [t]
    out = []
    for i, p in enumerate(NL_RE.split(t)):
        if i % 2 == 0:
            core = p.strip()
            if core and not unit_skip(core):
                out.append(core)
    return out


def compose(t, cache):
    """Ghép bản dịch của một chuỗi gốc từ cache (đơn vị nào chưa có thì giữ tiếng Anh)."""
    if not needs_split(t):
        return t if skip(t) else cache.get(t, t)
    parts = NL_RE.split(t)
    for i in range(0, len(parts), 2):
        p = parts[i]
        core = p.strip()
        if core and core in cache:
            lead = p[:len(p) - len(p.lstrip())]
            trail = p[len(p.rstrip()):]
            parts[i] = lead + cache[core] + trail
    return ''.join(parts)


def missing_units(t, cache):
    return [u for u in units_of(t) if u not in cache]


def cmd_extract(a):
    os.makedirs(WORK, exist_ok=True)
    for k in 'SDI':
        p = find_input(k)
        if not p:
            print(f'[!] thiếu file .{EXT[k]} trong thư mục {IN}/'); continue
        rows = read_table(p, HAS_LEN[k])
        with open(f'{WORK}/{k}.tsv', 'w', encoding='utf-8', newline='\n') as f:
            for sid, t in rows:
                f.write(f'{sid}\t{esc(t)}\n')
        todo = {u for _, t in rows for u in units_of(t)}
        print(f'{k}: {len(rows)} chuỗi, {len(todo)} đơn vị duy nhất cần dịch')


def load_tsv(k):
    rows = []
    for line in open(f'{WORK}/{k}.tsv', encoding='utf-8'):
        line = line.rstrip('\n')
        sid, _, t = line.partition('\t')
        rows.append((int(sid), unesc(t)))
    return rows


def load_cache():
    cache = {}
    cp = f'{WORK}/cache.jsonl'
    if os.path.exists(cp):
        for line in open(cp, encoding='utf-8'):
            if line.strip():
                o = json.loads(line); cache[o['en']] = o['vi']
    return cache


# ---------------------------------------------------------------- glossary
def load_glossary(path):
    ent = []
    if not path or not os.path.exists(path):
        return ent
    sec = ''
    for line in open(path, encoding='utf-8'):
        if re.match(r'^[A-E]\.\s', line):
            sec = line[0]
        if sec not in ('B', 'C', 'D') or line.startswith('--') or line.startswith('#'):
            continue
        for part in re.split(r"\s{2,}", line.strip()):
            m = re.match(r"^([^=]{2,60}?)\s=\s(.+)$", part)
            if m:
                en, vi = m.group(1).strip(), m.group(2).strip()
                for e in en.split(' / '):
                    if e.strip():
                        ent.append((e.strip(), vi))
    return ent


def pick_glossary(ent, batch_text):
    low = batch_text.lower()
    out = []
    for en, vi in ent:
        if re.search(r'(?<![A-Za-z])' + re.escape(en.lower()) + r'(?![A-Za-z])', low):
            out.append(f'{en} = {"(giữ nguyên)" if vi.startswith("(giữ") else vi}')
    return out[:60]


# ---------------------------------------------------------------- gọi model
SYSTEM = (
    "Bạn là dịch giả game chuyên nghiệp, dịch Fallout 4 từ tiếng Anh sang tiếng Việt tự nhiên, đúng chuẩn.\n"
    "QUY TẮC ĐỊNH DẠNG (bắt buộc):\n"
    "1. Mỗi dòng đầu vào có dạng `số|văn bản gốc`. Với MỖI dòng, trả về đúng MỘT dòng dạng `số|văn bản gốc ⇒ bản dịch`: "
    "chép lại nguyên văn bản gốc, rồi dấu ⇒, rồi bản dịch tiếng Việt. Đúng số dòng, đúng thứ tự, "
    "không bỏ sót, không gộp dòng, không giải thích.\n"
    "2. Giữ NGUYÊN các thẻ kỹ thuật và phím: <Alias=...>, <Global=...>, <Token...>, <mag>, <dur>, <cough>, "
    "[Activate], [Click], [Accept], [Cancel], [PipBoy], [VATS]..., %d, %s, %%, và dãy ký tự trang trí (====, :::, ---).\n"
    "3. Nhãn trạng thái trong ngoặc như [Lie], [Optional], [READY], [ACTIVE], [Deceased], [Cancelled] là chữ bình thường: "
    "dịch chữ bên trong, giữ dấu ngoặc.\n"
    "4. Giữ nguyên tên riêng, tên phe phái, địa danh, thuật ngữ game (Synth, Pip-Boy, VATS, Minutemen, Power Armor...).\n"
    "5. Nhiều dòng là nhãn menu / tên vật phẩm ngắn 1-3 từ (Silver, Remove, Weight, Exit...): phải dịch sang tiếng Việt theo nghĩa thông dụng trong game, "
    "chỉ giữ nguyên khi là tên riêng hoặc từ viết tắt.\n"
    "6. Chỉ viết bằng tiếng Việt (chữ Latin có dấu). TUYỆT ĐỐI không dùng tiếng Trung hay ngôn ngữ khác."
)


def load_guide(path):
    if path and os.path.exists(path):
        lines = [l.strip() for l in open(path, encoding='utf-8') if l.strip()]
        return '\n\nQUY TẮC VĂN PHONG:\n' + '\n'.join(f'- {l}' for l in lines)
    return ''


def call_ollama(model, system, user, host, num_ctx, think=False, temp=0.2):
    def post(payload):
        req = urllib.request.Request(host + '/api/chat', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())['message']['content']
    payload = {
        'model': model, 'stream': False, 'keep_alive': '30m',
        'options': {'temperature': temp, 'num_ctx': num_ctx},
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
    }
    if not think:
        try:
            return post({**payload, 'think': False})
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
    return post(payload)


def tags(t):
    out = []
    for m in TAG_RE.finditer(t):
        g = m.group(0)
        if g.startswith('[') and g[1:-1] not in KEYS and not KEY_PATTERN.fullmatch(g[1:-1]):
            continue  # nhãn chữ thường, không phải phím
        out.append(g)
    return sorted(out)


def reason(src, dst, ban_hostile=False):
    """Trả về None nếu bản dịch đạt, ngược lại trả về lý do (chuỗi ngắn)."""
    if dst is None or not dst.strip():
        return 'rỗng hoặc không đọc được'
    if tags(src) != tags(dst):
        return 'mất hoặc sai thẻ/phím/biến (phải giữ nguyên %d, %s, [Activate], <Alias=...>...)'
    if CJK_RE.search(dst):
        return 'có chữ Hán (chỉ được dùng tiếng Việt)'
    if '⇒' in dst or '\n' in dst or '\r' in dst or re.match(r'^\s*\d+\s*\|', dst):
        return 'sai định dạng'
    if len(src) >= 15 and (len(dst) > 3 * len(src) + 20 or len(dst) < 0.3 * len(src)):
        return 'độ dài bất thường'
    if ban_hostile and HOSTILE_RE.search(dst):
        return 'dùng "mày/tao" (thông báo hệ thống phải dùng "bạn")'
    if dst.strip() == src.strip() and len(re.findall(r'[A-Za-z]', src)) >= 3 and not (src.strip().isupper() and ' ' not in src.strip()):
        return 'giống hệt bản gốc'  # nếu là tên riêng thì giữ nguyên, nếu là từ thông thường thì phải dịch
    return None


def valid(src, dst, ban_hostile=False):
    return reason(src, dst, ban_hostile) is None


def norm(x):
    return re.sub(r'\W+', '', x.lower())


def echo_ok(src, echo):
    a, b = norm(src), norm(echo)
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.9


def parse_reply(reply, texts):
    """Trả về list bản dịch (None nếu thiếu hoặc nghi lệch dòng). Dòng echo phải khớp câu gốc."""
    n = len(texts)
    got = {}
    for line in reply.strip().splitlines():
        m = re.match(r'^\s*(\d+)\s*\|\s?(.*)$', line)
        if not m:
            continue
        idx, rest = int(m.group(1)), m.group(2)
        if '⇒' in rest:
            echo, _, vi = rest.partition('⇒')
            got[idx] = (echo.strip(), vi.strip())
        else:
            got[idx] = (None, rest.strip())
    out = []
    for i, src in enumerate(texts):
        g = got.get(i + 1)
        if g is None:
            out.append(None); continue
        echo, vi = g
        if n == 1:
            out.append(vi or None)  # 1 dòng thì không thể lệch
        elif echo is not None and (echo_ok(esc(src), echo) or echo_ok(src, echo)):
            out.append(vi or None)
        else:
            out.append(None)  # không chép lại hoặc chép sai -> nghi lệch dòng
    if n == 1 and out[0] is None and not got and reply.strip():
        out = [reply.strip().splitlines()[0].split('⇒')[-1].strip() or None]
    return out


def translate_batch(texts, glo, args, fake, hint='', temp=0.2):
    """texts: list[str]  ->  list[str|None]"""
    if fake:
        return ['VI: ' + t for t in texts]
    guide = load_guide(args.guide)
    gl = pick_glossary(glo, '\n'.join(texts))
    sysmsg = SYSTEM + guide + ('\n\nTHUẬT NGỮ BẮT BUỘC:\n' + '\n'.join(gl) if gl else '')
    user = '\n'.join(f'{i + 1}|{esc(t)}' for i, t in enumerate(texts))
    if hint:
        user += '\n\n(Lưu ý: ' + hint + ')'
    try:
        reply = call_ollama(args.model, sysmsg, user, args.host, args.num_ctx, args.think, temp)
    except Exception as e:
        print('  lỗi gọi Ollama:', e); time.sleep(3); return [None] * len(texts)
    return [unesc(r) if r is not None else None for r in parse_reply(reply, texts)]


def cmd_translate(a):
    glo = load_glossary(a.glossary)
    print(f'Glossary: {len(glo)} mục')
    cache = load_cache()
    keys = a.only.upper() if a.only else 'SDI'
    ban = set(keys) <= {'S', 'D'}  # UI/mô tả: không dùng "mày/tao"
    todo, seen = [], set()
    for k in keys:
        for _, t in load_tsv(k):
            for u in units_of(t):
                if u not in cache and u not in seen:
                    seen.add(u); todo.append(u)
    if a.limit:
        todo = todo[:a.limit]
    print(f'Cần dịch: {len(todo)} đơn vị (đã có {len(cache)} trong cache)')
    failed = open(f'{WORK}/failed.tsv', 'w', encoding='utf-8')
    out = open(f'{WORK}/cache.jsonl', 'a', encoding='utf-8')
    t0 = time.time(); done = 0

    def work(chunk):
        res = translate_batch(chunk, glo, a, a.fake)
        why = [reason(s_, r, ban) for s_, r in zip(chunk, res)]
        retry = [j for j, w in enumerate(why) if w]
        for j in retry:  # thử lại từng dòng riêng (1 dòng thì không thể lệch), kèm lý do lần trước
            for att, temp in enumerate((0.3, 0.6, 0.9)):
                hint = f'lần trước dịch không đạt vì: {why[j]}. Hãy dịch lại, chỉ dùng tiếng Việt.'
                r = translate_batch([chunk[j]], glo, a, a.fake, hint, temp)[0]
                w = reason(chunk[j], r, ban)
                res[j], why[j] = r, w
                if w is None:
                    break
            if why[j] == 'giống hệt bản gốc':
                why[j] = None  # thử lại vẫn giữ nguyên -> coi là tên riêng, giữ tiếng Anh
                res[j] = chunk[j]
        return chunk, res, why, len(retry)

    chunks = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for chunk, res, why, nretry in ex.map(work, chunks):
            for s_, r, w in zip(chunk, res, why):
                if w is not None:
                    failed.write(f"{esc(s_)}\t{w}\t{esc(r or '')}\n"); failed.flush()
                else:
                    out.write(json.dumps({'en': s_, 'vi': r}, ensure_ascii=False) + '\n')
            out.flush()
            done += len(chunk)
            el = time.time() - t0
            eta = el / done * (len(todo) - done) / 60
            print(f'  {done}/{len(todo)}  phải dịch lại riêng: {nretry}  ETA ~{eta:.0f} phút')
    print('Xong. Các dòng không đạt (kèm lý do và bản dịch cuối cùng của model) nằm trong work/failed.tsv, giữ tiếng Anh.')


def cmd_purge(a):
    """Xóa khỏi cache các dòng không đạt kiểm tra hiện tại để lần translate sau dịch lại.
    Mặc định giữ các dòng 'giống hệt bản gốc' (có thể là tên riêng); thêm --same để xóa cả chúng."""
    cache = load_cache()
    keys = a.only.upper() if a.only else 'SDI'
    ban = set(keys) <= {'S', 'D'}
    keep, drop = {}, 0
    for en, vi in cache.items():
        w = reason(en, vi, ban)
        if w is None or (w == 'giống hệt bản gốc' and not a.same):
            keep[en] = vi
        else:
            drop += 1
    with open(f'{WORK}/cache.jsonl', 'w', encoding='utf-8') as f:
        for en, vi in keep.items():
            f.write(json.dumps({'en': en, 'vi': vi}, ensure_ascii=False) + '\n')
    print(f'Giữ {len(keep)} dòng, xóa {drop} dòng không đạt (sẽ được dịch lại).')


def cmd_build(a):
    os.makedirs(OUT, exist_ok=True)
    cache = load_cache()
    for k in 'SDI':
        if not os.path.exists(f'{WORK}/{k}.tsv'):
            continue
        rows = load_tsv(k)
        new, miss = [], 0
        for sid, t in rows:
            if units_of(t) and missing_units(t, cache):
                miss += 1
            new.append((sid, compose(t, cache)))
        write_table(f'{OUT}/Fallout4_en.{EXT[k]}', new, HAS_LEN[k])
        print(f'{k}: ghi {len(new)} chuỗi, còn {miss} chuỗi chưa dịch đủ (phần thiếu giữ tiếng Anh)')


def cmd_check(a):
    for k in 'SDI':
        p = f'{OUT}/Fallout4_en.{EXT[k]}'
        src = find_input(k)
        if not (os.path.exists(p) and src):
            continue
        A, B = read_table(src, HAS_LEN[k]), read_table(p, HAS_LEN[k])
        assert [x[0] for x in A] == [x[0] for x in B], 'ID không khớp!'
        bad = same = 0
        for (_, s), (_, d) in zip(A, B):
            if not units_of(s):
                continue
            if s == d: same += 1
            if tags(s) != tags(d): bad += 1
        print(f'{k}: {len(A)} dòng, lệch thẻ/phím: {bad}, chưa dịch/giống gốc: {same}')


def cmd_selftest(a):
    os.makedirs(OUT, exist_ok=True)
    for k in 'SDI':
        p = find_input(k)
        if not p: continue
        rows = read_table(p, HAS_LEN[k])
        tmp = f'{OUT}/_selftest.{EXT[k]}'
        write_table(tmp, rows, HAS_LEN[k])
        same = read_table(tmp, HAS_LEN[k]) == rows
        ident = open(p, 'rb').read() == open(tmp, 'rb').read()
        print(f'{k}: đọc lại nội dung {"KHỚP" if same else "KHÔNG KHỚP"}; byte {"giống hệt" if ident else "khác (bình thường nếu gốc có ký tự Windows-1252, ghi ra UTF-8)"}')
        os.remove(tmp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('extract'); sp.add_parser('build'); sp.add_parser('check'); sp.add_parser('selftest')
    pg = sp.add_parser('purge'); pg.add_argument('--only', default=''); pg.add_argument('--same', action='store_true', help='xóa cả các dòng giống hệt bản gốc')
    t = sp.add_parser('translate')
    t.add_argument('--model', default='qwen3.5:9b')
    t.add_argument('--host', default='http://localhost:11434')
    t.add_argument('--glossary', default='Fallout4_VN_Glossary.txt')
    t.add_argument('--guide', default=GUIDE_FILE, help='file quy tắc văn phong ngắn (mặc định guide_local.txt)')
    t.add_argument('--batch', type=int, default=20)
    t.add_argument('--workers', type=int, default=1, help='số luồng song song (cần đặt OLLAMA_NUM_PARALLEL cùng giá trị)')
    t.add_argument('--num-ctx', type=int, default=4096, dest='num_ctx')
    t.add_argument('--think', action='store_true', help='bật chế độ suy nghĩ của model (mặc định TẮT cho nhanh)')
    t.add_argument('--only', default='')
    t.add_argument('--limit', type=int, default=0)
    t.add_argument('--fake', action='store_true', help='chạy thử luồng, không gọi model')
    a = ap.parse_args()
    {'extract': cmd_extract, 'translate': cmd_translate, 'build': cmd_build, 'purge': cmd_purge,
     'check': cmd_check, 'selftest': cmd_selftest}[a.cmd](a)
