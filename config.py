results_file = 'results.json'

B = 1000

tests = ['eq', 'eq1', 'dupeq1', 'eqx', 'dupeqx', 'ef', 'ef1', 'efx', 'po', 'eq+po', 'eq1+po', 'eqx+po', 'eq1+ef1+po', 'eqx+efx+po', 'ef+po', 'ef1+po', 'efx+po']
# props = ['eqx+po']

# solutions = ['market_eq', 'mnw', 'leximin']
solutions = ['minimax', 'leximin']
names = {'market_eq':'Market', 'mnw': 'MNW', 'leximin': 'Leximin',
         'minimax': 'MiniMax', 'mnw_binary': 'MNW', 'hefexistence': 'HEF-k',
         'shefexistence': 'uHEF-k', 'rr': 'RoundRobin', 'market': 'Alg-EF1+PO', 'envy_graph': 'EnvyGraph'}
