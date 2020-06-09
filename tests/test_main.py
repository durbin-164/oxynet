from oxynet.main import add, sub, mul

def test_add():
    assert add(100,200) == 300

def test_sub():
    assert sub(200, 100) == 100

def test_mul():
    assert mul(10,10) == 100