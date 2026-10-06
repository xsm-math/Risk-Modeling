"""One entry point: verified download, experiment and checked documentation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.credit.data import load_data
from src.credit.experiment import ROOT, run

if __name__=='__main__':
    load_data(ROOT/'data/raw/uci_credit.zip',download=True)
    run()
    from src.credit.reporting import write_readme_and_career
    write_readme_and_career(ROOT)
