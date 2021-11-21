"""Utility functions for text-based attacks. Adapted from https://github.com/thunlp/OpenAttack."""
def __measure(data, adversarial_sample, metrics):
    ret = {}
    for it in metrics:
        value = it.after_attack(data, adversarial_sample)
        if value is not None:
            ret[it.name] = value
    return ret


def __iter_dataset(dataset, metrics):
    for data in dataset:
        v = data
        for it in metrics:
            ret = it.before_attack(v)
            if ret is not None:
                v = ret
        yield v


def __iter_metrics(iterable_result, metrics):
    for data, result in iterable_result:
        adversarial_sample = result
        ret = {
            "data": data,
            "success": adversarial_sample is not None,
            "result": adversarial_sample,
            "metrics": {
                ** __measure(data, adversarial_sample, metrics)
            }
        }
        yield ret


def attack_process(attacker, victim, dataset, metrics):
    def result_iter():
        for data in __iter_dataset(dataset, metrics):
            yield attacker(victim, data)
    for ret in __iter_metrics(zip(dataset, result_iter()), metrics):
        yield ret
