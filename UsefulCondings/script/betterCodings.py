## Recursive function
def total(number: int) -> int:
	if number < 1:
		return 0
	return number + total(number-1)

# execute arbitrary number
def totalShow(number: int) -> int:
	print(total(number))