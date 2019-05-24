# ATTENTION! Remember to set B in config.py correctly

import time
import json
import logging
import numpy as np
from os.path import exists
from collections import Counter
from matplotlib import pyplot as plt

import config
from data import *
from solution import *
from property import *
from existence import *


def update_results(V, A, results, solution_type, run_time, instance_id=None, chores=False):
    tests = config.tests

    if not solution_type in results:
        results[solution_type] = dict()
        results[solution_type]['total_time'] = 0
        results[solution_type]['instances'] = 0
        for test in tests:
            results[solution_type][test] = 0
    for test in tests:
        # print('performing test {}'.format(test))
        props = test.rstrip().lstrip().split('+')
        propresults = list()
        for prop in props:
            propresults.append(int(check_prop(prop, V, A, chores=chores)))
        satisfies = int(np.sum(propresults) >= len(props))
        results[solution_type][test] += satisfies
        if not None == instance_id:
            logging.debug('instance {}, method {}, satisfies {}: {}'.format(instance_id, solution_type, test, satisfies))
    results[solution_type]['total_time'] += run_time
    results[solution_type]['instances'] += 1
    return results


def update_results_existence(V, exists, results, property_type, run_time, instance_id=None):
    if not property_type in results:
        results[property_type] = Counter()
    results[property_type]['exists']  += int(exists)
    results[property_type]['total_time'] += run_time
    results[property_type]['instances'] += 1
    if not None == instance_id:
        logging.debug('instance {}, property {}, exists {}'.format(instance_id, property_type, int(exists)))
    return results


def update_allocations(V, A, property_type, instance_id, allocations):
    if not instance_id in allocations:
        allocations[instance_id] = dict()
        allocations[instance_id]['valuations'] = V.tolist()
    allocations[instance_id][property_type] = A.tolist()
    return allocations


def eq_report(filename='results_spliddit.json'):
    properties = config.props
    solutions = config.solutions
    names = config.names
    with open(filename) as f:
        results = json.loads(f.read())
    instances = results['interesting_instances']
    for prop in properties:
        values = list()
        labels = list()
        exists = instances
        if prop in results:
            exists = results[prop]['exists']
        values.append(exists)
        labels.append('Existence')
        for sol in solutions:
            values.append(results[sol][prop])
            labels.append(names[sol])
        values = [100.*v/instances for v in values]
        x = np.arange(len(values))
        plt.bar(x, values, color='black')
        plt.ylim(0, 100)
        plt.xticks(x, labels, rotation=90)
        plt.title(prop.upper())
        plt.ylabel('%age of instances')
        plt.tight_layout()
        plt.savefig(filename + '_' + prop + '.png')
        plt.close('all')
    values = list()
    labels = list()
    for sol in solutions:
        values.append(1.*results[sol]['total_time']/results[sol]['instances'])
        labels.append(names[sol])
    x = np.arange(len(values))
    plt.bar(x, values, color='black')
    plt.xticks(x, labels, rotation=90)
    plt.title('Running time')
    plt.ylabel('Average running time (s)')
    plt.tight_layout()
    plt.savefig(filename + '_time' + '.png')
    plt.close('all')


def load_results(num_instances, filename=config.results_file):
    if not exists(filename):
        results = {'num_instances': num_instances, 'completed': 0, 'interesting_instances': 0}
        allocations = dict()
        return results, allocations
    with open(filename) as f:
        results = json.loads(f.read())
        allocations = dict()
        return results, allocations


def stats_spliddit():
    I = load_real_instances()
    num_instances = len(I)
    print('loaded {} instances'.format(num_instances))
    N = list()
    M = list()
    nm = Counter()
    for V in I:
        (n, m) = np.shape(V)
        if m < n or np.any(V == 0):
            continue
        N.append(n)
        M.append(m)
        nm[n, m] += 1
    print('N: ', min(N), max(N))
    print('M: ', min(M), max(M))
    print('3, 6: ', nm[3, 6])


def eq_goods():
    I = load_real_instances()
    num_instances = len(I)
    print('loaded {} instances'.format(num_instances))
    results, allocations = load_results(num_instances)
    count = results['completed']
    interesting_instances = results['interesting_instances']
    failed_instances = list()
    for V in I[count:]:
        print('working on instance ', count)
        count += 1
        (n, m) = np.shape(V)
        if m < n or np.any(V == 0):
            continue

        start = time.time()
        status, w, U, A = mnw(V)
        end = time.time()
        logging.info('done with {} for instance {} in {}s, status {}'.format('mnw', count, end - start, status))
        po = ispo(V, A)
        ef1 = isef1(V, A)
        if not po or not ef1:
            logging.debug('{} solution failed property test ef1={} po={}'.format('mnw', ef1, po))
            failed_instances.append((count, 'mnw'))
            with open('failed.json', 'w') as fo:
                fo.write(json.dumps(failed_instances, sort_keys=True, indent=4, separators=(',', ': ')))
            sys.exit()
        if status:
            results = update_results(V, A, results, 'mnw', end - start, instance_id=count)
        else:
            failed_instances.append((count, 'mnw'))

        start = time.time()
        status, w, U, A = leximin(V)
        end = time.time()
        logging.info('done with {} for instance {} in {}s, status {}'.format('leximin', count, end - start, status))
        po = ispo(V, A)
        eqx = iseqx(V, A)
        if not po or not eqx:
            logging.debug('{} solution failed property test eqx={} po={}'.format('leximin', eqx, po))
            failed_instances.append((count, 'leximin'))
            with open('failed.json', 'w') as fo:
                fo.write(json.dumps(failed_instances, sort_keys=True, indent=4, separators=(',', ': ')))
            sys.exit()
        if status:
            results = update_results(V, A, results, 'leximin', end - start, instance_id=count)
        else:
            failed_instances.append((count, 'leximin'))

        start = time.time()
        status, A, p = market_eq(V)
        end = time.time()
        logging.info('done with {} for instance {} in {}s, status {}'.format('market_eq', count, end - start, status))
        po = ispo(V, A)
        eq1 = iseq1(V, A)
        if not po or not eq1:
            logging.debug('{} solution failed property test eq1={} po={}'.format('market_eq', eq1, po))
            failed_instances.append((count, 'market_eq'))
            with open('failed.json', 'w') as fo:
                fo.write(json.dumps(failed_instances, sort_keys=True, indent=4, separators=(',', ': ')))
            sys.exit()
        if status:
            results = update_results(V, A, results, 'market_eq', end - start, instance_id=count)
        else:
            failed_instances.append((count, 'market_eq'))

        results['completed'] = count
        interesting_instances = interesting_instances + 1
        results['interesting_instances'] = interesting_instances

        with open(config.results_file, 'w') as fo:
            fo.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(config.results_file+'.allocs', 'w') as fo:
            fo.write(json.dumps(allocations, sort_keys=True, indent=4, separators=(',', ': ')))
        with open('failed.json', 'w') as fo:
            fo.write(json.dumps(failed_instances, sort_keys=True, indent=4, separators=(',', ': ')))
    return None
