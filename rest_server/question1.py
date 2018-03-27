from random import randint
def solution(x):
	temp = x
	#print(temp)
	inPlaceQuickSort(temp,0,len(temp)-1)
	return temp

def inPlaceQuickSort(A,start,end):
	if start<end:
		pivot=randint(start,end)
		temp=A[end]
		A[end]=A[pivot]
		A[pivot]=temp
		p=inPlacePartition(A,start,end)
		inPlaceQuickSort(A,start,p-1)
		inPlaceQuickSort(A,p+1,end)


def inPlacePartition(A,start,end):
	pivot=randint(start,end)
	temp=A[end]
	A[end]=A[pivot]
	A[pivot]=temp
	newPivotIndex=start-1
	for index in range(start,end):
		if A[index]<A[end]:#check if current val is less than pivot value
			newPivotIndex=newPivotIndex+1
			temp=A[newPivotIndex]
			A[newPivotIndex]=A[index]
			A[index]=temp
	temp=A[newPivotIndex+1]
	A[newPivotIndex+1]=A[end]
	A[end]=temp
	return newPivotIndex+1