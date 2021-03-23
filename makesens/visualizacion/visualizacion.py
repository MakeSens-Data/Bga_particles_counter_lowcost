import matplotlib.pyplot as plt

def vizualize (variables:list,namevariable:list,xl:str,yl:str,font:int,figsize:tuple):
    plt.figure(figsize = figsize)
    for i in range(0,len(variables)):
        variables[i].plot(label = namevariable[i])
    plt.gcf().autofmt_xdate()
    plt.xlabel(xl,fontsize=font)
    plt.ylabel(yl,fontsize=font)
    plt.legend(fontsize =font-3)
    plt.show()