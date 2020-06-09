from oxynet.main import add, sub

def test_add():
    assert add(100,200) == 300

def test_sub():
    assert sub(200, 100) == 100
