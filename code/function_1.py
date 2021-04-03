# Flatten Function
flat_list = []
def flatten_list(my_list):
    """
    Değiken olarak liste alır.
    Liste içerisindeki iç içe olan non-scalar and scalar değişkenleri tek 
    bir liste içerisinde yazar.(Flatten)
    Geriye liste döndürür.
    """ 
    if type(my_list) is list:
        
        for li in my_list:
            if type(li) is list:
                flatten_list(li)
                
            elif type(li)  is tuple:
                flat_list.extend(li)
                
            elif type(li)  is set:
                flat_list.extend(li)
                
            elif type(li)  is dict:
                flat_list.extend(li)
                flat_list.extend(li.values())
                
            else:
                flat_list.append(li)
    else: 
        return print("Variable is not list")
        
    return flat_list         
