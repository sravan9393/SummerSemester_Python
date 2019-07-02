class Solution:
    
    def lengthOfLongestSubstring(self, s: str) -> int:
    
        # prechecking the len to be atleast 1 ( --> Doing it in BigOh(n) single parse<-- )
        if len(s):
            result = []
            
            temp = s[0]
            for i in s[1:]:
                if i not in temp:
                    temp += i
                elif i == temp[0]:
                    result.append(temp)
                    temp = temp[1:] + i
                    
                elif i == temp[-1]:
                    result.append(temp)
                    temp = i
                else:
                    result.append(temp)
                    temp = temp[temp.find(i) + 1:] + i
            
            result.append(temp)
            
            # getting the max length of the longest substring
            max_len = len(max(result, key=len))
            
            # printing all the longest substrings
            list_1 = []
            for substr in result:
                if max_len == len(substr):
                    list_1.append((substr, max_len))
        
            print(list_1)

if __name__ == '__main__':
    
    # Taking string as input and printing the longest substrings from the string
    Sol = Solution()
    string = input('Enter the string:')
    Sol.lengthOfLongestSubstring(s=string)            