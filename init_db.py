import pymysql
import sys

def init_db():
    try:
        # Connect to MySQL without database first
        con = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            charset='utf8'
        )
        
        with con.cursor() as cur:
            # Create Database
            cur.execute("CREATE DATABASE IF NOT EXISTS crop")
            cur.execute("USE crop")
            
            # Create Signup Table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS signup (
                username VARCHAR(50) PRIMARY KEY,
                password VARCHAR(50),
                contact_no VARCHAR(15),
                email VARCHAR(50),
                address VARCHAR(250)
            )
            """
            cur.execute(create_table_query)
            con.commit()
            
        con.close()
        print("[AgriRAG] MySQL Database and Tables verified successfully.")
    except Exception as e:
        print(f"[WARNING] MySQL Initialization failed: {e}")
        print("Please ensure MySQL is running on 127.0.0.1:3306 with user 'root' and password 'root'.")

if __name__ == "__main__":
    init_db()
    sys.exit(0)
