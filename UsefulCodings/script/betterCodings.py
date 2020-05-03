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
		history[number][target] = True
		return True
		# return history[number][target] := True # for python 3.8

	# not choice vec[number-1]
	if subsetSum(number-1, target, vec, history): 
		history[number][target] = True
		return True

	history[number][target] = False
	return False

def subsetSumShow(number: int, target: int, vec: list) -> str:
	# save result list
	history = [[0 for i in range(target+1)] for j in range(number+1)]
	if subsetSum(number, target, vec, history): print('yes man')
	else: print('no man')

#### logging function
# from logging import getLogger