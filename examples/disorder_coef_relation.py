import DrunkardWalkSD as DWSD
import numpy as np
import matplotlib.pyplot as plt

coin_disorders = np.linspace(0.1, 1, 200)
alpha_list = []

quantity_sidewalks = 10_000
size_sidewalks = 500

city = DWSD.City(quantity_sidewalks, size_sidewalks)

for w in coin_disorders:
    city.set_coin_W(w)
    city.roam()
    alpha = city.make_std_graph(tail=100, loglog=True, only_coef=True)
    alpha_list.append(alpha)
    city.reset_data()
    
quad_coef, lin_coef, ind_term = np.polyfit(coin_disorders, alpha_list, 2)

plt.title(f"W vs Alpha")
plt.xlabel("W")
plt.ylabel("Ang. Coef.")

plt.scatter(coin_disorders, alpha_list)
plt.plot(coin_disorders, np.polyval([quad_coef, lin_coef, ind_term], coin_disorders),
         label=f"a = {quad_coef:.4f}, b = {lin_coef:.4f}", color='r', linewidth=2)

plt.legend(loc='upper right')

plt.savefig("comparison_alpha_w.png")