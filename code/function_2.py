#Reverse Function
def get_the_opposite(my_list):
    """
    Değişken olarak liste alır.
    Liste elemanlarını tersten yazdırır. Liste içerisindeki 
    non-scaler ve scaler değişkenler de dahil olmak üzere. 
    Geriye liste döndürür.
    """
    reverse_list = []
    if type(my_list) is list:
        my_list = my_list[::-1]
        
        for li in my_list: 
            if type(li) is list: 
                reverse_list.append(li[::-1])

            elif type(li) is dict:
                li = list(li.items())
                for l in li[::-1]:
                    reverse_list.append(l[::-1])
                    
            elif type(li) is set or tuple:
                li = list(li)    
                reverse_list.append(li[::-1])
                    
    else:
        return print("Variable is not list")
    
    return reverse_list
