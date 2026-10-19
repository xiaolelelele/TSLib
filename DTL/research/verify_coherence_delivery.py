"""Check final sources and deliverables without changing scientific results."""
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
WORKSPACE = RESEARCH.parents[1]
OUT = RESEARCH / 'analysis/coherence_validation_20261008'
RUN = WORKSPACE / 'DTL/trend_lab_runs/time_trajectory_20260930'
DECK = RESEARCH / 'presentations/2026-10-08'
checks = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(name, valid):
    checks.append({'name': name, 'pass': bool(valid)})


def hashes(items, base, tag):
    for name, expected in items.items():
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        check(f'{tag}/{name}', path.exists() and digest(path) == expected)


lock = read(OUT/'controls_rule_lock.json')
hashes(lock['source_sha256'], RUN, 'original_frozen_source')
check('main_controls_protocol', digest(WORKSPACE/lock['protocol']) == lock['protocol_sha256'])
check('main_controls_code', digest(RESEARCH/'validate_direction_mean_coherence.py') == lock['script_sha256'])
scope = read(OUT/'supplement_scope.json')
hashes(scope['inputs_sha256'], WORKSPACE, 'supplement_inputs')
check('supplement_code', digest(RESEARCH/'summarize_direction_mean_coherence.py') == scope['code_sha256'])
flow = read(OUT/'real_flow_certificates.json')
hashes(flow['source_hashes'], WORKSPACE, 'flow_inputs')
hashes(flow['code_hashes'], OUT, 'flow_code')
floor = read(OUT/'moment_floor_validation/rule_lock.json')
hashes(floor['source_sha256'], WORKSPACE, 'floor_inputs')
check('floor_code', digest(RESEARCH/'probe_whole_horizon_moment_floor.py') == floor['code_sha256'])

assertions = {}
for name in ['audit_inputs.json', 'audit_flow.json', 'theory_results.json',
             'audit_controls.json', 'audit_witnesses.json', 'audit_moment_floor.json']:
    data = read(OUT/name)
    check(name+'/no_failure', data['failure_count'] == 0)
    assertions[name] = data['check_count']
completed = read(OUT/'controls_completion.json')
check('no_new_fit_or_selection', completed['model_fits'] == 0 and not completed['new_hyperparameter_selection'])
check('legacy_not_confirmation', completed['legacy_test_is_confirmation'] is False)
check('24_bitexact_original_replays', completed['original_replays_bit_exact'] == 24)
check('floor_validation_only_posthoc', floor['posthoc'] and floor['validation_only'])

main = RESEARCH/'trend_accuracy_literature_and_ideas_2026-09-26.md'
before = (OUT/'main_document_before.md').read_text().split('## 0.', 1)[1].rstrip()
after = main.read_text().split('## 0.', 1)[1].split('## 14.', 1)[0].rstrip()
check('historical_sections_0_through_13_unchanged', before == after)
report = RESEARCH/'coherence_validation_results_2026-10-08.md'
for match in re.finditer(r'\]\((/home/[^)]+)\)', report.read_text()):
    path = re.sub(r':\d+$', '', match.group(1))
    check('report_link/'+path, Path(path).exists())

manifest = read(DECK/'sources_manifest.json')
for name, entry in manifest['sources'].items():
    check('ppt_source/'+name, digest(entry['path']) == entry['sha256'])
check('ppt_builder', digest(DECK/'build_research_deck.py') == manifest['script_sha256'])
quality = read(DECK/'export_quality_audit.json')
ppt = DECK/'趋势预测_文献创新与验证_2026-10-08.pptx'
pdf = ppt.with_suffix('.pdf')
check('ppt_sha', digest(ppt) == quality['pptx_sha256'])
check('pdf_sha', digest(pdf) == quality['pdf_sha256'])
check('ppt_checks_clear', not quality['missing_pdf_text'] and not quality['out_of_bounds'] and not quality['source_drift'])
with zipfile.ZipFile(ppt) as archive:
    names = archive.namelist()
    slides = [n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
    notes = [n for n in names if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', n)]
    charts = [n for n in names if re.fullmatch(r'ppt/charts/chart\d+\.xml', n)]
    check('26_editable_slides', len(slides) == 26)
    check('26_speaker_notes', len(notes) == 26)
    check('native_charts', len(charts) == 3)
    check('no_flattened_slide_images', all(b'<p:pic>' not in archive.read(n) for n in slides))
    check('primary_links', quality['primary_link_relationships'] >= 30)
pdf_info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True, text=True).stdout
check('26_pdf_pages', re.search(r'^Pages:\s+26\s*$', pdf_info, re.M) is not None)
plot = read(OUT/'plot_render_manifest.json')
check('plot_saved_estimates_unchanged', digest(plot['source']) == plot['source_sha256'])
check('plot_output_sha', digest(plot['output']) == plot['output_sha256'])

result = {'scope': 'Final artifact/source checks only, no fitting, selection or scientific rescoring.',
          'check_count': len(checks), 'failure_count': sum(not c['pass'] for c in checks),
          'engineering_assertions': assertions, 'total_engineering_assertions': sum(assertions.values()),
          'checks': checks,
          'root_visual_inspection': 'All 26 slides via two contact sheets; full slides 21/23 plus paired-difference plot. No clipping; plot legend/ticks moved outside data.'}
(OUT/'delivery_checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
files = {p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json'}
files.update({p for p in DECK.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
files.update({main, report, RESEARCH/'coherence_validation_protocol_2026-10-08.md',
              RESEARCH/'validate_direction_mean_coherence.py', RESEARCH/'summarize_direction_mean_coherence.py',
              RESEARCH/'probe_whole_horizon_moment_floor.py', RESEARCH/'render_coherence_effects.py', Path(__file__)})
(OUT/'delivery_manifest.json').write_text(json.dumps({
    'client_date': '2026-10-08', 'scope': 'Final research and presentation artifacts; hashes are provenance, not new scientific evidence.',
    'file_count': len(files), 'files_sha256': {str(p): digest(p) for p in sorted(files)},
}, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'root_visual_inspection')}, ensure_ascii=False))
if result['failure_count']:
    raise SystemExit('Final delivery checks failed; see delivery_checks.json')
