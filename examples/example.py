import DrunkardWalkSD as DWSD

quantity_sidewalks = 10_000
size_sidewalks = 501
max_steps = 500
disorder_intensity = 1.0

city = DWSD.City(quantity_sidewalks, size_sidewalks, disorder_intensity)

city.roam(max_steps)

city.make_avg_graph()
city.make_std_graph()