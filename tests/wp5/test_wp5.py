import sys

from subprojects.wp5 import file_saving, hihi


####### test section #######################################################################
def test_hihi():
    assert hihi() == "hihihi"


def test_file_saving():
    assert file_saving(42) == "hat funktioniert: 42"


####### example how to call functions from the outside (not relevant for tests) #############
if __name__ == "__main__":
    print(sys.path)
    print(hihi())
    print(file_saving(42))
