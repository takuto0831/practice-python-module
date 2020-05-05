#### Recursive function (with type annotation)
# factorial function
def total(number: int) -> int:
  if number < 1:
    return 0
  return number + total(number-1)

# execute arbitrary number
def totalShow(number: int) -> int:
  print(total(number))


# recursive function with dynamic programming 
def dynamicFibonacci(number: int, history: list) -> int:
  # base case
  if number == 0: return 0;
  elif number == 1: return 1;

  # check already calcurated?
  if history[number] >= 0: 
    return history[number];
  history[number] = dynamicFibonacci(number-1,history) + dynamicFibonacci(number-2,history)
  print(str(number) + ':' + str(history))
  return history[number]

def dynamicFibonacciShow(number: int, history: list) -> int:
  # logger = getLogger("Fibonacci tes:")
  print(dynamicFibonacci(number, history))
  

# recursive function with subset sum problem, dynamic programming
def subsetSum(number: int, target: int, vec: list, history: list) -> bool:
  # base case
  if number == 0:
    if target == 0:
      return True;
    else:
      return False;

  # check already calcurated
  if history[number][target] != 0: return history[number][target]

  # choice vec[number-1]
  if subsetSum(number-1, target - vec[number-1], vec, history): 
    history[number][target] = True; return True
    # return history[number][target] := True # for python 3.8?

  # not choice vec[number-1]
  if subsetSum(number-1, target, vec, history): 
    history[number][target] = True; return True
    # return history[number][target] := True # for python 3.8?

  history[number][target] = False; return False
  # return history[number][target] := False

def subsetSumShow(number: int, target: int, vec: list) -> str:
  # save result list
  history = [[0 for i in range(target+1)] for j in range(number+1)]
  if subsetSum(number, target, vec, history): print('yes man')
  else: print('no man')

# recursive function with dfs (depth-first search)
def dfsShow(vec: list):
	if len(vec) == 10:
		print(vec)
		return
	for num in range(2):
		# back track
		vec.append(num) # add new number
		dfs(vec) # call next..
		vec.pop() # remove new number

#### logging function
from logging import getLogger, StreamHandler, Formatter, DEBUG

def loggerTest():
  # logger object
  logger = getLogger("Log Test")
  # set logger log level
  logger.setLevel(DEBUG)

  # set handler
  streamHandler = StreamHandler()
  # set handler log level
  streamHandler.setLevel(DEBUG)
  # output format
  handlerFormat = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  streamHandler.setFormatter(handlerFormat)

  # set handler for logger
  logger.addHandler(streamHandler)
  logger.debug("Hello world")
  logger.info("yahho")
  logger.warning("don't miss it")


#### python technics for Data scientist
# generator (near Class ...): return value/keep state/not keep all list
def countUp():
  x=0
  while True:
    yield x
    x += 1

def countUpShow(number:int, func):
  for Iter in func:
    print(Iter)
    if Iter == number:
      break

# itertools
def itertoolShow():
	import itertools
	print(list(itertools.combinations_with_replacement(range(5), 3)))

# enumerate function
def enumerateShow():
  data = ['りんご', 'バナナ', 'みかん', 'ぶどう']
  for Iter, value in enumerate(data):
    print('{}番目: {}'.format(Iter, value))

# map and filter
def mapAndFilterShow():
  data = [1,5,6,7,10,32,34,45,1,2]
  print(list(map(str,data)));
  print(list(filter(lambda x: x > 10, data)))

# object oriented (with document)
def objectTotalShow():
  '''
  calcurate sum of two numbers
  
  parameter
  num1 (int, float)
  num2 (int, float)
  
  return
  int or float
  '''

  from typing import TypeVar
  Number = TypeVar('Number', int, float)
  def total(num1: Number, num2: Number) -> Number:
    return num1 + num2

  # some test
  print(total(1,2));
  print(total(1.5,2.3));
  print(total(1,2.5));
  print(total("rtsat","tar")); # it not show error ....
  













