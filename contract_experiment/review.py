"""Offline human-review view. Never changes reviews, exclusions, or model inputs."""

import difflib
import html
from pathlib import Path

from .storage import digest, read_csv, read_json


def escape(value):
    return html.escape(str(value), quote=True)


def evidence(doc, annotation):
    pieces = []
    for sid in annotation["spans"]:
        start, end = doc["spans"][sid]
        pieces.append(f'<div class="evidence"><small>span {sid} · chars [{start}, {end})</small>'
                      f'<pre>{escape(doc["text"][start:end])}</pre></div>')
    return "".join(pieces) or '<span class="muted">Gold Evidence 없음</span>'


def label_table(left, right, hypotheses, only_conflicts=False):
    rows = []
    for hid, hypothesis in hypotheses.items():
        a, b = left["annotations"][hid], right["annotations"][hid]
        different = a["choice"] != b["choice"]
        if only_conflicts and not different:
            continue
        style = ' class="conflict"' if different else ""
        rows.append(f'<tr{style}><td><strong>{escape(hid)}</strong><p>{escape(hypothesis)}</p></td>'
                    f'<td><strong>{escape(a["choice"])}</strong>{evidence(left, a)}</td>'
                    f'<td><strong>{escape(b["choice"])}</strong>{evidence(right, b)}</td></tr>')
    if not rows:
        return '<p class="same">가설별 Gold 라벨이 모두 같습니다.</p>'
    return ('<div class="scroll"><table class="labels"><thead><tr><th>가설</th>'
            f'<th>{escape(left["key"])} · 라벨 / 근거</th><th>{escape(right["key"])} · 라벨 / 근거</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')


def pair_section(pair, left, right, hypotheses, review, anchor):
    raw_same = left["text"] == right["text"]
    kinds = {"raw_exact": "원문 완전 일치", "normalized_exact": "정규화 후 일치", "similar": "유사 중복 후보"}
    a_lines, b_lines = left["text"].splitlines(), right["text"].splitlines()
    changes = [operation for operation in difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).get_opcodes()
               if operation[0] != "equal"]
    changed_a = sum(end - start for _, start, end, _, _ in changes)
    changed_b = sum(end - start for _, _, _, start, end in changes)
    if raw_same:
        diff = '<p class="same">두 원문은 문자 단위로 완전히 같습니다. 아래에서 원문과 라벨을 확인하세요.</p>'
    else:
        diff = '<div class="scroll">' + difflib.HtmlDiff(wrapcolumn=100).make_table(
            a_lines, b_lines, fromdesc=escape(left["key"]), todesc=escape(right["key"]),
            context=True, numlines=3) + '</div>'
    notes = review.get("notes", "")
    decision = review.get("decision", "") or "미입력"
    conflicts = len(pair["label_conflicts"])
    label_summary = (f'<p class="danger">Gold 라벨 불일치: {conflicts}개 가설. 라벨을 고르거나 고칠 필요는 없습니다.</p>'
                     if conflicts else '<p class="same">Gold 라벨 불일치 없음.</p>')
    return f'''<section class="pair" id="{anchor}" data-required="{str(pair['review_required']).lower()}">
<h2>{escape(pair['left'])} ↔ {escape(pair['right'])}</h2>
<div class="badges"><span>{escape(kinds.get(pair['kind'], pair['kind']))}</span>
<span>5단어 묶음 Jaccard: {pair['similarity']:.6f} ({pair['similarity']:.2%})</span>
<span>검토 필수: {'예' if pair['review_required'] else '아니오 · Train 교차 보고'}</span></div>
<p>원문 길이: {escape(left['key'])} {len(left['text']):,}자 / {escape(right['key'])} {len(right['text']):,}자.
줄 단위로 바뀐 영역: 왼쪽 {changed_a}줄 / 오른쪽 {changed_b}줄. 줄바꿈 차이도 포함합니다.</p>
<p>현재 CSV 결정: <strong>{escape(decision)}</strong><br>메모: {escape(notes) or '미입력'}</p>
<h3>① 원문에서 다른 부분</h3>
<p class="muted">바뀐 부분과 주변 3줄을 표시합니다. 왼쪽은 원문 A, 오른쪽은 원문 B입니다.
<span class="diff_sub">삭제</span> · <span class="diff_add">추가</span> · <span class="diff_chg">변경</span>.
단어 줄바꿈은 화면 표시용이며 저장된 계약서를 수정하지 않습니다.</p>
{diff}
<details><summary>전체 원문 나란히 보기</summary><div class="sources">
<div><h4>{escape(left['key'])}</h4><pre>{escape(left['text'])}</pre></div>
<div><h4>{escape(right['key'])}</h4><pre>{escape(right['text'])}</pre></div></div></details>
<h3>② 라벨이 다른 가설과 양쪽 Gold Evidence</h3>{label_summary}
{label_table(left, right, hypotheses, only_conflicts=True) if conflicts else ''}
<p>가설별 근거 텍스트 일치: {'예' if pair['evidence_text_equal'] else '아니오 · 아래 근거를 확인'}.
근거 표기 차이와 정답 라벨 충돌은 별개입니다.</p>
<details><summary>17개 가설 전체의 라벨·근거 보기</summary>{label_table(left, right, hypotheses)}</details>
<h3>③ 검토 기록</h3><p>위 내용을 확인한 뒤 <code>review.csv</code>의
<code>{escape(pair['pair_id'])}</code> 행에 <code>decision</code>과 <code>notes</code>를 입력하세요.
이 보고서는 읽기 전용이며 CSV를 변경하지 않습니다.</p>
<p><a href="#top">목록으로 돌아가기</a></p></section>'''


def write_review_report(audit_path, output=None):
    audit_path = Path(audit_path)
    audit = read_json(audit_path)
    claimed_digest = audit.get("audit_digest")
    if digest({key: value for key, value in audit.items() if key != "audit_digest"}) != claimed_digest:
        raise ValueError("Audit file changed. Use the unchanged audit.json for the report.")
    payload = read_json(audit_path.parent / "candidate_documents.json")
    documents = {doc["key"]: doc for doc in payload["documents"]}
    review_rows = read_csv(audit_path.parent / "review.csv")
    reviews = {row["pair_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows) or set(reviews) != {p["pair_id"] for p in audit["pairs"]}:
        raise ValueError("Review rows do not match audit candidates.")
    required_keys = {key for pair in audit["pairs"] for key in (pair["left"], pair["right"])}
    if required_keys - set(documents):
        raise ValueError("Missing candidate documents. Use candidate_documents.json from this audit.")
    sections, links = [], []
    for number, pair in enumerate(audit["pairs"], 1):
        anchor = f"pair-{number}"
        left, right = documents[pair["left"]], documents[pair["right"]]
        conflicts = [hid for hid in payload["hypotheses"] if left["annotations"][hid]["choice"] != right["annotations"][hid]["choice"]]
        if set(conflicts) != set(pair["label_conflicts"]):
            raise ValueError(f"Candidate labels do not match audit: {pair['pair_id']}")
        sections.append(pair_section(pair, left, right, payload["hypotheses"], reviews[pair["pair_id"]], anchor))
        links.append(f'<li><a href="#{anchor}">{escape(pair["pair_id"])}</a> · {pair["similarity"]:.6f}'
                     f' · 라벨 불일치 {len(conflicts)}개 · {"검토 필수" if pair["review_required"] else "Train 교차 보고"}</li>')
    body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ContractNLI 중복 검토</title>
<style>
body{{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#182238;margin:0}}
main{{max-width:1500px;margin:auto;padding:24px}}header,.pair{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:24px;margin:0 0 24px}}
h1,h2,h3{{line-height:1.35}}a{{color:#125db4}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;margin:8px 0}}
.muted,small{{color:#58677d}}.same{{color:#136344}}.danger{{color:#a51e32;font-weight:600}}
.badges{{display:flex;gap:8px;flex-wrap:wrap}}.badges span{{background:#edf2f8;border-radius:6px;padding:4px 10px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce3ec;padding:8px;text-align:left;vertical-align:top}}
table.diff{{font:13px/1.5 ui-monospace,monospace}}table.diff td{{white-space:pre-wrap;min-width:24px}}
.diff_add{{background:#d4f4df}}.diff_sub{{background:#ffe0e0}}.diff_chg{{background:#fff0b3}}
.diff_header{{background:#eef2f8}}.diff_next{{background:#edf2f8}}.conflict{{background:#fff5f3}}
.sources{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.sources>div{{min-width:0}}
.labels td:first-child{{width:30%}}.evidence{{border-top:1px solid #e3e8ef;margin-top:8px;padding-top:4px}}
summary{{cursor:pointer;color:#125db4;font-weight:600;padding:10px 0}}code{{background:#edf2f8;padding:2px 5px;border-radius:4px}}
@media(max-width:800px){{main{{padding:12px}}header,.pair{{padding:16px}}.sources{{grid-template-columns:1fr}}}}
</style></head><body><main><header id="top"><h1>ContractNLI 중복 후보 검토</h1>
<p>후보 {len(audit['pairs'])}쌍 · 사람 검토 필수 {sum(p['review_required'] for p in audit['pairs'])}쌍.
인터넷이나 모델 실행 없이 열 수 있는 비교 자료입니다.</p>
<p><strong>similarity는 원문 문자의 일치율이 아닙니다.</strong> 정규화한 연속 5단어 묶음 집합의 Jaccard 점수입니다.
예를 들어 0.98을 ‘계약 내용의 98%가 같다’로 해석하지 말고, 아래 실제 변경 내용을 확인하세요.</p>
<p><code>raw_exact</code>: 원문 완전 일치 · <code>normalized_exact</code>: 정규화 후 일치 · <code>similar</code>: 유사 후보.</p>
<p>같은 계약서로 확인되면 CSV에 <code>duplicate</code>, 별개 계약서면 <code>distinct</code>와 판단 근거를 적습니다.
같은 내용인데 라벨이 충돌하는 Dev/Test 그룹은 prepare에서 전체 제외합니다. Train 교차 후보는 보고만 합니다.
라벨을 직접 선택하거나 수정하지 않습니다.</p>
<p>Gold Evidence는 중복을 검토하는 사람에게 제공하는 자료이며 압축 프롬프트에는 들어가지 않습니다.</p>
<ul>{''.join(links) or '<li>후보 없음</li>'}</ul></header>{''.join(sections)}</main></body></html>'''
    path = Path(output) if output is not None else audit_path.parent / "review_report.html"
    if path.suffix.lower() != ".html" or path.resolve() in {audit_path.resolve(), (audit_path.parent / "review.csv").resolve(), (audit_path.parent / "candidate_documents.json").resolve()}:
        raise ValueError("Choose an .html output path; audit/review/source files cannot be replaced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"Review report: {path.resolve()}")
    return path
