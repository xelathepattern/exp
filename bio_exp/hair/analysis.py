import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import csv

# Read data from file
data_file = "data.txt"
key_file = "name_key.csv" # maps real names to IDs
redact = False

dtype = {'names': ('name', 'width'), 'formats': ('S32', 'i4')}
data = np.loadtxt(data_file, dtype=dtype, delimiter=',')
data = np.sort(data, order='width')

if os.path.exists(key_file):
    with open(key_file) as f:
        key = {row[0].strip(): row[1].strip() for row in csv.reader(f)}
    data["name"] = [key.get(n.decode().strip(), n.decode().strip()).encode() for n in data["name"]]
    bar_plot_save_file = "plots/bar_plot.png"
else:
    bar_plot_save_file = "plots/bar_plot_anon.png"

mean = np.mean(data['width'])
std = np.std(data['width'])

# Plot
plt.figure(figsize=(10, 5))
plt.bar(data["name"], data["width"], color='skyblue')
plt.xticks(rotation=45, ha="right")
plt.ylabel("Width (microns)")
plt.title("Hair")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.plot(data["name"], np.full_like(data["width"], fill_value=mean), color='black', linestyle='--')
plt.text(0, mean, f"mean = {mean:.1f}")
plt.plot(data["name"], np.full_like(data["width"], fill_value=mean+std), color='grey', linestyle='--')
plt.text(0, mean+std, f"std = {std}")
plt.plot(data["name"], np.full_like(data["width"], fill_value=mean-std), color="grey", linestyle='--')

plt.savefig(bar_plot_save_file)


plt.figure()
sp.stats.probplot(data['width'], plot=plt)
plt.savefig("plots/normal_qq.png")

plt.figure()
plt.scatter(np.full_like(data["width"], fill_value=0), data["width"])
plt.savefig("plots/scatterplot.png")

edf = sp.stats.ecdf(data["width"]) 

fig, ax = plt.subplots()
edf.cdf.plot(ax)
lower, upper = edf.cdf.confidence_interval()
lower_vals = lower.probabilities
upper_vals = upper.probabilities
mask = np.isfinite(lower_vals) & np.isfinite(upper_vals)
ax.step(edf.cdf.quantiles, edf.cdf.probabilities, where="post", label="ECDF", color="blue")
ax.fill_between(edf.cdf.quantiles[mask], lower_vals[mask], upper_vals[mask], step="post", color="gray", alpha=.3, label="95% CI")
#ax.plot(edf.quantiles, cdf_bounds, color="green"
#CDF of unit normal is Phi(z) = 1/2 (1 + erf(z/sqrt(2)))
xs = np.linspace(data["width"][0], data["width"][-1], len(data)*20)
ax.plot(xs, 1/2 * (1 + sp.special.erf((xs - mean) / std) / (2**.5)), color="green")
fig.savefig("plots/edf.png")

plt.show()
