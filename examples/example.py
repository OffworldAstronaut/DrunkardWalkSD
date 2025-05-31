import DrunkardWalkSD as DWSD

quantity_sidewalks = 10_000
size_sidewalks = 100
max_steps = 1_000

city = DWSD.City(quantity_sidewalks, size_sidewalks, 0.5)

city.roam(max_steps)

city.make_avg_graph()
city.make_std_graph()