import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    """
    Connects to the default PostgreSQL server and creates the cardio_db database.
    """
    # ⚠️ Update these if your local PostgreSQL uses a different password or username
    user = 'postgres'
    password = 'hybesty123'
    host = 'localhost'
    port = '5432'
    db_name = 'cardio_db'

    print(f"Connecting to PostgreSQL server at {host}:{port}...")
    try:
        # Connect to the default 'postgres' database to issue the CREATE DATABASE command
        connection = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres"
        )
        
        # We must set isolation level to AUTOCOMMIT to create a database
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = connection.cursor()
        
        # Check if the database already exists to avoid errors
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f'CREATE DATABASE {db_name}')
            print(f"✅ Database '{db_name}' created successfully!")
        else:
            print(f"ℹ️ Database '{db_name}' already exists.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please make sure your PostgreSQL server is running and your password is correct.")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()
            print("PostgreSQL connection closed.")

if __name__ == "__main__":
    create_database()
