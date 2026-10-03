from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from schedulers import cscan, fcfs, scan, sstf


COURSE_QUEUE = [98, 183, 37, 122, 14, 124, 65, 67]


@pytest.mark.parametrize(("scheduler", "expected"), [(fcfs, 640), (sstf, 236), (scan, 236), (cscan, 386)])
def test_course_example(scheduler, expected):
    assert scheduler(COURSE_QUEUE, 53).total_seek == expected


def test_scan_reaches_physical_zero():
    assert scan([40, 60], 50).total_seek == 110


def test_cscan_counts_wrap():
    assert cscan([40, 60], 50).total_seek == 388


def test_duplicate_requests_are_preserved():
    result = sstf([50, 50, 49], 50)
    assert result.order == [50, 50, 49]
    assert result.total_seek == 1


def test_invalid_track_rejected():
    with pytest.raises(ValueError):
        fcfs([200], 50)

