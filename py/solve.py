import sys
import time

class solver():
    def __init__ (self):
        pass

    def _same_row(self, i,j): return (i//9 == j//9)
    def _same_col(self, i,j): return (i-j) % 9 == 0
    def _same_block(self, i,j): return (i//27 == j//27 and i%9//3 == j%9//3)

    def solve(self, input):
    
        if len(input) != 81:
            print('wrong number of inputs')
            return
        
        if 0 not in input:
            if self.validate(input):
                print('result: ')
                
            else:
                print('No result: ')
                
            self.disp(input)
            # print("Time used : " + str(time.time() - start))
            sys.exit()
        else:
            i = input.index(0)
            excluded_numbers = set()
            for j in range(81):
                if self._same_row(i,j) or self._same_col(i,j) or self._same_block(i,j):
                    excluded_numbers.add(input[j])

            for m in range(1, 10):
                if m not in excluded_numbers:
                    self.solve(input[:i] + [m] + input[i+1:])
            
    def disp(self, m):
        for i in range(9):
            print(m[i*9: (i+1)*9])
                 
    def validate(self, input):
        # validate no empty
        n = set(range(1, 10))
        re = True;
        re = re and (0 not in input)
        if not re:
            print('Zero found')
            return re
        for i in set(range(0, 9)):
            # All lines 
            re = re and (n == {j for j in input[i * 9 : i * 9 + 9]})
            if not re:
                print('Line ' + str(i) + ' invalid')
                return re
            # All columns
            re = re and (n == {input[j*9 + i]  for j in range(0, 9)})
            if not re:
                print('Column ' + str(i) + ' invalid')
                return re
        for i in [0, 3, 6, 27, 30, 33, 54, 57, 60]:
            # All blocks
            re = re and (n == set(input[i:i+3] + input[i+9:i+12] + input[i+18:i+21]))
            if not re:
                print('Block of ' + str(i) + ' invalid')
                return re
        if re:
            print('result valid')
            return re
        
    def disp_result(self):
        print(self.result)


        
        
        
if __name__ == '__main__':
    '''
    input1 = [5, 3, 0, 0, 7, 0, 0, 0, 0, 6, 0, 0, 1, 9, 5, 0, 0, 0, 0, 9, 8, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0, 6, 0, 0, 0, 3, 4, 0, 0, 8, 0, 3, 0, 0, 1, 7, 0, 0, 0, 2, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0, 2, 8, 0, 0, 0, 0, 4, 1, 9, 0, 0, 5, 0, 0, 0, 0, 8, 0, 0, 7, 9]
    output = [5, 3, 4, 6, 7, 8, 9, 1, 2, 6, 7, 2, 1, 9, 5, 3, 4, 8, 1, 9, 8, 3, 4, 2, 5, 6, 7, 8, 5, 9, 7, 6, 1, 4, 2, 3, 4, 2, 6, 8, 5, 3, 7, 9, 1, 7, 1, 3, 9, 2, 4, 8, 5, 6, 9, 6, 1, 5, 3, 7, 2, 8, 4, 2, 8, 7, 4, 1, 9, 6, 3, 5, 3, 4, 5, 2, 8, 6, 1, 7, 9]
    '''

    # input1 = [1, 0, 0, 0, 7, 0, 0, 3, 0, 8, 3, 0, 6, 0, 0, 0, 0, 0, 0, 0, 2, 9, 0, 0, 6, 0, 8, 6, 0, 0, 0, 0, 4, 9, 0, 7, 0, 9, 0, 0, 0, 0, 0, 5, 0, 3, 0, 7, 5, 0, 0, 0, 0, 4, 2, 0, 3, 0, 0, 9, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4, 3, 0, 4, 0, 0, 8, 0, 0, 0, 9]
    # input2 = [1, 6, 9, 2, 7, 0, 4, 3, 0, 8, 3, 4, 6, 0, 0, 7, 9, 0, 0, 0, 2, 9, 4, 3, 6, 1, 8, 6, 0, 0, 0, 3, 4, 9, 0, 7, 4, 9, 8, 0, 0, 0, 3, 5, 1, 3, 0, 7, 5, 9, 0, 0, 6, 4, 2, 0, 3, 4, 5, 9, 1, 0, 6, 9, 0, 0, 0, 0, 2, 0, 4, 3, 0, 4, 0, 3, 8, 0, 0, 0, 9]
    # input3 = [1, 6, 9, 8, 7, 0, 4, 3, 0, 8, 3, 4, 6, 0, 0, 7, 9, 0, 0, 0, 2, 9, 4, 3, 6, 1, 8, 6, 0, 0, 0, 3, 4, 9, 0, 7, 4, 9, 8, 0, 0, 0, 3, 5, 1, 3, 0, 7, 5, 9, 0, 0, 6, 4, 2, 0, 3, 4, 5, 9, 1, 0, 6, 9, 0, 0, 0, 0, 2, 0, 4, 3, 0, 4, 0, 3, 8, 0, 0, 0, 9]
    # solver = solver()
    # start = time.time()
    # a = solver.solve(input1)



