import numpy as np
import pandas as pd
import streamlit as st
#import seaborn as sns
import streamlit as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

filest = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def objective(x, a, b, c, wct):
        fun = (a*x**2 + b*x + c) * (1-wct)
        return -fun

def objectives(x, a, b, c, wct):
    fun = (a*x**2 + b*x + c) * (1-wct)
    return fun

def objective_qt(x, a, b, c):
    fun = (a*x**2 + b*x + c)
    return -fun

def objectives_qt(x, a, b, c):
    fun = (a*x**2 + b*x + c)
    return fun

def constraint(x):
    return 1000 - x

def Average(lst):
    return sum(lst) / len(lst)

def rsq(x, y, degree):
    results = {}

    correlation = np.corrcoef(x, y)[0,1]
    #print(f"corr: {correlation}")

    # r
    results['correlation'] = correlation
    # r-squared
    results['determination'] = correlation**2

    return rsq_list.append(results['determination'])

def mainkan(uploaded_file):
    cons = [{"type": "ineq", "fun": constraint}]

    #Polynomial Loop
    #sl1=[]
    #sl2=[]
    #itc=[]
    rsq_list=[]
    rates=[]
    avg=[]
    avg2=[]

    #days = [8, 10, 12, 14]
    days=[8]

    strt = 8
    ed = 30
    #for f in range(strt, ed):
    #print("====================== 2nd Order Single Well Optimization =============================")
    st.subheader("2nd Order Single Well Optimization")
    for f in days:
        #print(strt)
        condition=[]
        yopro=[]
        ytpro=[]
        glpro=[]
        yopr=[]
        ytpr=[]
        ytpr_copy=[]
        yopr_copy=[]
        sl1=[]
        sl2=[]
        itc=[]
        #print("====================== GLIR Optimized, Days:", f, "=============================")
        st.subheader("GLIR Optimized, Days:")
        #fd = pd.read_csv("wellgabung.csv")
        #fd = pd.read_csv("welledit.csv")
        fd = pd.read_csv(uploaded_file)
        #fd.drop(7, inplace=True)
        for ij in range(0, len(fd)-f+1+1):
            #fd = pd.read_csv("wellgabung.csv")
            #fd = pd.read_csv("welledit.csv")
            fd = pd.read_csv(uploaded_file)
            df = fd.iloc[ij:ij+f]
            #print(df)

            # create some random data
            x = df["GLIR"].values
            y = df["Qt"].values
            #y = df["qt_lc"].values
            w = df["wc"].values
            
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0)

            model = np.polyfit(x, y, 2)
            #print(model)
            slope1 = model[0]
            slope2 = model[1]
            intercept = model[2]
            
            if slope1 > 0:
                #y_pred1 = -slope1 * x ** 2 + slope2 * x - intercept
                
                if slope2 > 0 and intercept > 0: # + +
                    y_pred1 = -slope1 * x ** 2 - slope2 * x - intercept
                    par = (-slope1, -slope2, -intercept, w[-1])
                    condition.append("c1")
                    #print("c1:", [-slope1, -slope2, -intercept])

                elif slope2 > 0 and intercept < 0: # + -
                    y_pred1 = -slope1 * x ** 2 - slope2 * x + intercept
                    par = (-slope1, -slope2, +intercept, w[-1])
                    #print("c2:", [-slope1, -slope2, +intercept])
                    condition.append("c2")

                elif slope2 < 0 and intercept > 0: # - +
                    y_pred1 = -slope1 * x ** 2 - slope2 * x + intercept
                    par = (-slope1, -slope2, intercept, w[-1])
                    #print("c3:", [-slope1, -slope2, +intercept])
                    condition.append("c3")

                #elif slope2 < 0 and intercept < 0:
                else: # - -
                    y_pred1 = -slope1 * x ** 2 - slope2 * x - intercept
                    par = (-slope1, -slope2, -intercept, w[-1])
                    #print("c4:", [-slope1, -slope2, -intercept])
                    condition.append("c4")
            #Bring
            else:
                """y_pred1 = slope1 * x ** 2 + slope2 * x + intercept
                par = (slope1, slope2, intercept, w[-1])
                print("c5:", [slope1, slope2, intercept])
                condition.append("c5")"""
                
                if slope2 > 0 and intercept > 0: # + +
                    y_pred1 = -slope1 * x ** 2 + slope2 * x - intercept
                    par = (-slope1, +slope2, -intercept, w[-1])
                    condition.append("c5")
                    #print("c5:", [slope1, slope2, intercept])

                elif slope2 > 0 and intercept < 0: # + -
                    y_pred1 = -slope1 * x ** 2 + slope2 * x - intercept
                    par = (-slope1, slope2, -intercept, w[-1])
                    #print("c6:", [slope1, slope2, +intercept])
                    condition.append("c6")

                elif slope2 < 0 and intercept > 0: # - +
                    y_pred1 = -slope1 * x ** 2 - slope2 * x + intercept
                    par = (-slope1, -slope2, intercept, w[-1])
                    #print("c7:", [-slope1, -slope2, +intercept])
                    condition.append("c7")

                #elif slope2 < 0 and intercept < 0:
                else: # - -
                    y_pred1 = -slope1 * x ** 2 - slope2 * x - intercept
                    par = (-slope1, -slope2, -intercept, w[-1])
                    #print("c8:", [-slope1, -slope2, -intercept])
                    condition.append("c8")
            
            
                
            """if slope2 > 0 and intercept > 0:
                y_pred1 = -slope1 * x ** 2 + slope2 * x + intercept

            elif slope2 > 0 and intercept < 0:
                y_pred1 = -slope1 * x ** 2 + slope2 * x - intercept

            elif slope2 < 0 and intercept > 0:
                y_pred1 = -slope1 * x ** 2 - slope2 * x + intercept

            elif slope2 < 0 and intercept < 0:
                y_pred1 = -slope1 * x ** 2 - slope2 * x - intercept"""
            
            """print("x:", x)
            print("y:", y_pred1)
            print("sl1:", slope1)
            print("sl2:", slope2)
            print("itc:", intercept)"""
                    
            
            #print(y_pred1)
            
            mymodel = np.poly1d(np.polyfit(x, y_pred1, 2))
            myline = np.linspace(min(x), max(x), len(x))
            
            #print(x)

            #print(f"nilai a = {slope1}, nilai b = {slope2}, nilai c = {intercept}")    
            #boun = (0, 1000)
            #boun = (min(x), max(x))
            boun = (0, 1000)
            #boun = (min([min(x), min(x2)]), max([max(x), max(x2)]))
            #b1 = (min(x), max(x))
            #b1 = (0.75*x[-2], 1.25*x[-2])
            b1 = (0.75*x[-2], 1*x[-2])
            #b2 = (min(x2), max(x2))
            #bound = [boun, boun]
            bound = [b1]
            #solx = [int(sol.x[0]), int(sol.x[1])]s


            x0 = x[-1]

            #par = (slope1, slope2, intercept, w[-1])
            #print(par)
            par_qt = (slope1, slope2, intercept)

            if slope1<0:
                sol = minimize(objective,x0,args=par,method='SLSQP',bounds=bound,constraints=cons)
                #print(f"nilai qo prediksi {objectives(sol.x,slope1,slope2,intercept,w[-1])}")
                po = objectives(sol.x,slope1,slope2,intercept,w[-1])[0]
                yopro.append(po)
                #yopro_copy.append(po)
                glpro.append(int(sol.x[0]))
                #glpro_copy.append(int(sol.x[0]))
            else:
                sol = minimize(objectives,x0,args=par,method='SLSQP',bounds=bound,constraints=cons)
                #print(f"nilai qo prediksi {objectives(sol.x,slope1,slope2,intercept,w[-1])}")
                po = objectives(sol.x,slope1,slope2,intercept,w[-1])[0]
                yopro.append(po)
                #yopro_copy.append(po)
                glpro.append(int(sol.x[0]))
                #glpro_copy.append(int(sol.x[0]))

            ytpr.append(y_pred1[f-1])
            ytpr_copy.append(y_pred1[f-1])
            sl1.append(slope1)
            sl2.append(slope2)
            itc.append(intercept)

            ij+=1
            
            #plt.scatter(x,y,label=f"{ij}",alpha=0.4)
            #plt.plot(x,y_pred1)
            """plt.plot(myline, mymodel(myline), color="orange")
            plt.scatter(x,y_pred1, color="r", label="Qt Pred")
            plt.scatter(x,y,alpha=1, label="Qt Data")
            #sns.regplot(x,y_pred1)
            #plt.legend(loc=2)
            plt.xlabel("GLIR (MSCFD)")
            plt.ylabel("Qt (BFPD)")
            plt.grid()
            #plt.title(f"Qt Regression Curve vs Qt Plot Data, Iteration:{ij+1}, Day:{ij+1}-{ij+8}")
            plt.title(f"Qt Regression Curve vs Qt Plot Data, Iter:{ij-1}")"""
            
            #print("x:", x)
            #print("y:", y_pred1)
            #print("sl1:", slope1)
            #print("sl2:", slope2)
            #print("itc:", intercept)
            
            #plt.grid()
        #plt.show()
            
        #plt.grid()
        #plt.show()

        objectives(sol.x,slope1,slope2,intercept,w[-1])[0]

        s=0
        for i in range(f-1, len(fd)):
            qo=(sl1[s] * fd["GLIR"][i] * fd["GLIR"][i] + sl2[s]*fd["GLIR"][i] + itc[s])*(1 - fd["wc"][i])
            s+=1
            yopr.append(qo)
            yopr_copy.append(qo)
        
        yo_s=[]
        for i in fd.loc[f-1:,"Qo"]:
            yo_s.append(i)
            
        scs=[]
        scs_avg=[]
        scs_pm=[]
        for i in range(0, len(fd)-f+1):
            if yo_s[i] <= yopro[i]:
                scs.append(abs(yopro[i]-yo_s[i]))
                scs_avg.append(yopro[i]-yo_s[i])
                scs_pm.append("+")
            else:
                scs.append(abs(yopro[i]-yo_s[i]))
                scs_avg.append(yopro[i]-yo_s[i])
                scs_pm.append("-")

        scs_r=[]
        for i in scs_avg:
            if i > 0:
                scs_r.append(i)
        #print("Rate:", len(scs_r)/(len(fd)-f+1))
        #print(len(scs_r))
        #print(len(fd)-f+1)
        rate = len(scs_r)/(len(fd)-f+1)
        rates.append(rate)
        
        #print(len(fd["Qo"][f-1:]))
        #print(len(yopro_copy))
        rsq(fd["Qo"][f-2:], yopro, 2)
        #print("rsq:", rsq_list)

        #print(yopro)
        #print(yopr)
        #print(fd["Date"][f-1:])
        #print(len(ytpr_copy))
        #print(len(ytpro_copy))
        
        """dict1 = {"Date":fd["Date"][f-1:], "wc":fd["wc"][f-1:]}
        dr = pd.DataFrame(dict1)
        dr["GLIR"] = fd["GLIR"][f-1:]
        dr["GLIR Opt"] = glpro[1:]
        yo_s=[]
        for i in fd.loc[f-1:,"Qo"]:
            yo_s.append(i)
        
        dr["Success"] = scs_pm
        dr["Qo"] = fd["Qo"][f-1:]
        #dr["Qo Pred"] = yopr
        dr["Qo Opt"] = yopro[1:]
        dr["dQo"] = scs
        dr["Cond."] = condition[1:]
        dr["dGLIR"] = dr["GLIR"] - dr["GLIR Opt"]"""
        
        #dr.to_csv("well_new_QoWcQt.csv")

        #st.subheader('Data Full')
        #st.write(dr)
        #print(dr)
        average = Average(scs)
        avg.append(average)
        average_inc=Average(scs_avg)
        avg2.append(average_inc)
        #print("AVG", average)
        #print(" ")
        st.subheader('Details')
        st.write("Flowrate Increase Percentage:", rate*100, "%")
        st.write("Average Qo Difference (bbl/d):", average)
        st.write("Average Qo Increase (bbl/d):", average_inc)
        """print("Flowrate Increase Percentage:", rate*100, "%")
        print("Average Qo Difference (bbl/d):", average)
        print("Average Qo Increase (bbl/d):", average_inc)"""
        #print("Rate:", rate)
        #print("")
        
        """dict11 = {"Date":fd["Date"][f-1:]}
        dr = pd.DataFrame(dict11)
        dr["Qo"] = fd["Qo"][f-1:]
        dr["wc"] = fd["wc"][f-1:]
        dr["Qt"] = fd["Qt"][f-1:]
        #print(dr)
        #print("")
        st.subheader('Data Full')
        st.write(dr)
        
        
        
        dict111 = {"Date":fd["Date"][f-1:]}
        dr = pd.DataFrame(dict111)
        dr["GLIR"] = fd["GLIR"][f-1:]
        dr["GLIR_pred"] = glpro[1:]
        #print(dr)
        st.subheader('Data Full')
        st.write(dr)"""
        
        #82
        
        #arr = [i for i in range(f-1, len(fd))]
        """plt.plot(arr, yopro[1:], color="red", label="Qo Opt")
        plt.plot(arr, fd["Qo"][f-1:len(fd)], color="blue", label="Qo Data")
        plt.title(f"Qo Data vs Qo Optimized, Days: {f}")
        plt.ylabel("Qo (BOPD)")
        plt.grid()
        plt.xlabel("Day")
        plt.legend()
        plt.show()"""
        
        """plt.plot(arr, glpro[1:], color="red", label="GLIR Opt", linewidth=9)
        plt.plot(arr, fd["GLIR"][f-1:len(fd)], color="blue", label="GLIR Data")
        plt.title(f"GLIR Data vs GLIR Optimized, Days: {f}")
        plt.ylabel("GLIR (MSFCD)")
        plt.grid()
        plt.xlabel("Day")
        plt.legend()
        plt.show()"""

mainkan(filest)
