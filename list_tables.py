import sys
import os
sys.path.append(os.getcwd())

try:
    from mp_db_io import DataIO
    io = DataIO()
    db = io.db
    from sqlalchemy import create_engine, inspect
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"]
        )
    )
    insp = inspect(engine)
    tables = [t for t in insp.get_table_names() if 'SegmentHelper' in t]
    for t in tables:
        print(t)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
