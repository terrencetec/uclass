"""
"""
import uclass.helloworld
import uclass.clitools


def test_helloworld():
    string = uclass.helloworld.helloworlds(1)
    assert string == 'Hello World!'
