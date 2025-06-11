import DrunkardWalkSD as DWSD

quantity_sidewalks = 10_000
size_sidewalks = 500
disorder_intensity = 0.0

city = DWSD.City(quantity_sidewalks, size_sidewalks, disorder_intensity)

city.roam()

#city.make_avg_graph(plot_only=False)
city.make_std_graph(tail=100, plot_only=False, loglog=True)
#city.make_endpos_graph()