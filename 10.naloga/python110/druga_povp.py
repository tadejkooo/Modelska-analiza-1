import matplotlib.pyplot as plt
import numpy as np

y=[]


f = open('data/signal0.dat', 'r',encoding='utf-8')
data_file=f.readlines()

for vrsta in data_file:
    # print(i)
    # print(type(vrsta))
    # print(vrsta)
    y.append(float(vrsta))

frek2=np.fft.fft(y)
x=[]
x_original=[]

f = open('data/signal2.dat', 'r',encoding='utf-8')
data_file=f.readlines()

i=0
for vrsta in data_file:
    x.append(float(vrsta))
    x_original.append(float(vrsta))
    i=i+1


frekvencno=np.fft.fft(x)
f=abs(frekvencno)**2

N=len(frekvencno)
print(len(frekvencno))


# g=np.array(y)-np.array(x)
# plt.plot(x)
# plt.title('absolutna napaka')
# plt.show()



##blocno povprecje:
dolzina=5
for i in range(dolzina,N-dolzina):
    vsota=0
    for j in range(-dolzina,dolzina):
        vsota+=x[i+j]
    x[i]=vsota/dolzina/2

vsota1=0
vsota2=0
for i in range(dolzina):
    vsota1=x_original[i]
    vsota2=x_original[-i]

for i in range(dolzina):
    x[i]=vsota1/dolzina
    x[-i-1]=vsota2/dolzina



f1=np.fft.fft(x)
f2=np.fft.fft(x_original)
f3=np.fft.fft(y)

plt.plot(f1)
plt.plot(f3)
plt.legend(['blocno','pravilno'])
plt.show()
# plt.plot(x_original)
plt.plot(x)
plt.plot(y)
# plt.legend(['original','blocno','no noise'])
plt.show()

plt.plot(f)
plt.yscale('log')
plt.show()

##fi:
mejna=50
vsota=0
for i in range(mejna,N):
    vsota+=f1[i]
povprecje=vsota/(N-mejna)
print(povprecje)

sum_1=np.ones(len(frekvencno))*povprecje

##prenosna funkcija:
fi=[]
tau=16
r=[]

for i in range(len(frekvencno)):
    fi.append(f[i]/(f[i]+sum_1[i]))
    r.append(np.exp(-i/tau)/(2*tau))
for i in range(int(len(frekvencno)/2)):
    r[-i]=r[i]
# print(fi)

R=np.fft.fft(r)

##inverzna fft:

invert=np.fft.ifft(f1/R)
invert2=np.fft.ifft(f3/R)
invert3=np.fft.ifft(frek2/R*fi)


plt.plot(invert2, 'tab:blue')
#plt.plot(invert, 'tab:red')
plt.plot(invert3, 'tab:green')
plt.legend(['signal0','signal2 (bločno povprečenje)'])
plt.xlabel('t')
plt.ylabel(r'$u(t)$')
# plt.savefig('12_2_blocno_popvprecenje.png')
# plt.legend(['sum_blocno_povprecje','original data'])
plt.show()

    

