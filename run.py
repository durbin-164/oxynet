import argparse
from subprocess import call

REPORT_DIR = "reports/"

def _run_coverage_pytest(run_type, out_dir):
    type_str = "--cov-report="+ run_type + ":" + out_dir
    exit(call(["pytest", "--cov=oxynet", "tests/", type_str]))


def __setup__():
    call(["pip", "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('script_type', help='setup | coverage/xml')
    parser.add_argument('--out', help='output file path')
    args = parser.parse_args()

    script_type = args.script_type

    if script_type == 'setup':
        __setup__()

    elif script_type == 'test':
        exit(call('pytest', 'tests/'))

    elif script_type == 'coverage/xml':
        out_dir = args.out if args.out else REPORT_DIR + 'coverage/coverage.xml'
        _run_coverage_pytest('xml', out_dir)