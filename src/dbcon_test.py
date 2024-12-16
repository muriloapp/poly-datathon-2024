import psycopg

conn = psycopg.connect("postgresql://janos:(q0)r:0|QL>N[}b>>PA<6Eu6fE5x@database-3.cluster-czkmkismw47i.us-west-2.rds.amazonaws.com/datathon_db")

c = conn.cursor()

print(c.execute("SELECT now()").fetchall())

