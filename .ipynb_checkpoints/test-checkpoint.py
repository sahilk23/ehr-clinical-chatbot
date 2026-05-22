import duckdb

# connect to the database file
con = duckdb.connect("mimic_iv_demo.duckdb")

# list all tables
print(con.execute("SHOW TABLES").fetchall())