"""Run Adjustment Blend's tests in Maya's Python.

    mayapy tests/run.py                every group
    mayapy tests/run.py blend          the named groups
    mayapy tests/run.py --list         the group names

Prints one line per check, "PASS name: detail", "FAIL name: detail" or "KNOWN name: detail", and
exits with 1 when any check fails. KNOWN is a check for a bug that is on record and not fixed yet:
it does not fail the run. When such a check starts to pass it is reported as a FAIL, so that its
known_bug marker gets removed.

The tests import the adjustment_blend package this file sits in, wherever that folder is. They
build their scenes in code (tests/scenes.py), so they need no scene files.

To add a group, write tests/test_<group>.py with run(report) and list it in GROUPS.
report(ok, name, detail) records one check: ok is True, False, or None for a known bug.
"""
import argparse
import importlib
import logging
import os
import sys
import time
import traceback

GROUPS = [  # group name, module
    ("blend", "adjustment_blend.tests.test_blend"),
    ("layers", "adjustment_blend.tests.test_layers"),
    ("keys", "adjustment_blend.tests.test_keys"),
]


def main():
    parser = argparse.ArgumentParser(description="Run Adjustment Blend's tests in mayapy.")
    parser.add_argument("groups", nargs="*", help="groups to run; all of them when none is named")
    parser.add_argument("--list", action="store_true", help="print the group names and stop")
    options = parser.parse_args()
    names = [name for name, _module in GROUPS]
    if options.list:
        print("\n".join(names))
        return 0
    unknown = [g for g in options.groups if g not in names]
    if unknown:
        parser.error("no such group: %s. The groups are %s" % (", ".join(unknown), ", ".join(names)))
    wanted = options.groups or names

    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(package) != "adjustment_blend":
        print("FAIL run: the package folder must be named adjustment_blend, and this one is %s" % package)
        return 1
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.dirname(package))
    # A crash in Maya must not open the error report window on the user's screen.
    os.environ.setdefault("MAYA_DISABLE_CER", "1")
    import maya.standalone
    maya.standalone.initialize(name="python")
    import adjustment_blend
    if not os.path.samefile(os.path.dirname(os.path.abspath(adjustment_blend.__file__)), package):
        print("FAIL run: imported adjustment_blend from %s, not from %s" % (adjustment_blend.__file__, package))
        return 1
    logging.getLogger("adjustment_blend").setLevel(logging.WARNING)

    counts = {True: 0, False: 0, None: 0}

    def report(ok, name, detail=""):
        counts[ok] += 1
        word = {True: "PASS", False: "FAIL", None: "KNOWN"}[ok]
        print("%s %s%s" % (word, name, ": " + detail if detail else ""), flush=True)

    start = time.perf_counter()
    for name, module_name in GROUPS:
        if name not in wanted:
            continue
        group_start = time.perf_counter()
        try:
            importlib.import_module(module_name).run(report)
        except Exception:
            report(False, name, "the group stopped with an error")
            traceback.print_exc(file=sys.stdout)
        print("     %s took %.1f seconds" % (name, time.perf_counter() - group_start), flush=True)
    print("%d checks: %d passed, %d failed, %d known bugs, in %.1f seconds" % (
        sum(counts.values()), counts[True], counts[False], counts[None], time.perf_counter() - start), flush=True)
    return 1 if counts[False] else 0


def _quit(code):
    """End the process without Maya's shutdown, which can crash, hang or change the exit code. os._exit still
    unloads Maya's libraries, and a crash there leaves a recovered scene in the temp folder on every run."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess(ctypes.c_void_p(kernel.GetCurrentProcess()), ctypes.c_uint(code))
    os._exit(code)


if __name__ == "__main__":
    _quit(main())
