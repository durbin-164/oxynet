from oxynet.main import add, sub, mul, div,pow

def test_add():
    assert add(100,200) == 300

def test_sub():
    assert sub(200, 100) == 100

def test_mul():
    assert mul(10,10) == 100

def test_div():
    assert div(100, 10) == 10

def test_pow():
    assert pow(2,2) ==4