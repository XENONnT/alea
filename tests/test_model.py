from alea import StatisticalModel


def test_statistical_model():
    try:
        error_raised = True
        StatisticalModel()
        error_raised = False
    except Exception:
        print('Error correctly raised when directly instantiating StatisticalModel')
    else:
        if not error_raised:
            raise RuntimeError(
                'Should raise error when directly instantiating StatisticalModel')
