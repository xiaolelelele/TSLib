"""Check hand-written report cells and local Markdown links against artifacts."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def main():
    research = Path(__file__).resolve().parent
    report = research/"recent_data_rolling_results_2026-09-30.md"
    analysis = research/"analysis/recent_rolling_20260930"
    summary = pd.read_csv(analysis/"summary/summary.csv")
    physical = pd.read_csv(analysis/"summary/physical_ranges_posthoc.csv")
    physical = physical[physical.method=="base_clipped"].groupby(["dataset", "year", "backbone"]).mean(numeric_only=True)
    incremental = pd.read_csv(analysis/"summary/clipped_increment_vs_clipped_base.csv")
    incremental = incremental[incremental.method=="raw_selected_clipped"].groupby(["dataset", "year", "backbone"]).mean(numeric_only=True)
    lines = report.read_text().splitlines()
    tables, current = [], []
    for line in lines+[""]:
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current=[]
    checks=0
    datasets={"SMARD":"smard_de", "USCRN":"uscrn_crossville"}
    backbones={"DLinear":"dlinear", "iTransformer":"itransformer"}
    def numbers(cells):
        return [float(v.strip().replace("−", "-").replace(",", "")) for v in cells]
    for line in tables[0][2:]:
        cells=[v.strip() for v in line.strip("|").split("|")]
        data,year=cells[0].split("/")
        path=research.parent/"trend_lab_runs/recent_rolling_20260930/results_v2"/datasets[data]/year/"dataset.json"
        metadata=json.loads(path.read_text())
        expected=[metadata["splits"][key]["kept"] for key in ["train","calibration","riskfit","validation","test"]]
        assert numbers(cells[1:])==expected
        checks+=5
    for index in [1,2]:
        table=tables[index]
        for line in table[2:]:
            cells=[v.strip() for v in line.strip("|").split("|")]
            data,year,backbone=cells[0].split("/")
            key=(datasets[data],int(year),backbones[backbone])
            measured=numbers(cells[1:5])
            if index==1:
                f=summary[(summary.dataset==key[0])&(summary.year==key[1])&(summary.backbone==key[2])]
                raw=f[f.method=="raw_selected"].iloc[0]
                gate=f[f.method=="paired_gate"].iloc[0]
                expected=[raw.raw_gain_pp,raw.mse_change_pct,gate.raw_gain_pp,gate.mse_change_pct]
                assert raw.joint_wins==3 and gate.joint_wins==3 and cells[-1]=="3/3"
            else:
                raw=physical.loc[key]
                extra=incremental.loc[key]
                expected=[raw.raw_gain_pp,raw.mse_change_pct,extra.raw_gain_pp,extra.mse_change_pct]
            assert np.allclose(measured,expected,atol=.000050001,rtol=0),(key,measured,expected)
            checks+=4
    broken=[]
    docs=[report,research/"recent_dataset_catalog_2026-09-30.md",research/"recent_prior_art_addendum_2026-09-30.md",
          research/"recent_data_rolling_protocol_2026-09-30.md",analysis/"reproduction.md"]
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)",doc.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path=(doc.parent/target.split("#",1)[0]).resolve()
            checks+=1
            if not path.exists():
                broken.append({"document":str(doc),"target":target})
    gate=pd.read_csv(analysis/"summary/gate_increment_vs_alpha1.csv")
    assert len(gate)==24 and not gate.joint_win.any()
    result={"report_table_numeric_cells":84,"checks":checks,"broken_local_links":broken,
            "gate_joint_wins_vs_same_candidate":int(gate.joint_win.sum()),"status":"pass" if not broken else "fail"}
    (analysis/"final_document_checks.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert not broken


if __name__=="__main__":
    main()
