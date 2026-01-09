import pandas as pd
from sqlalchemy import create_engine, text

DB_NAME = "miniproject"
DB_URL_BASE = "mysql+pymysql://root:12341234@localhost:3306"

# -------------------------
# 1. DB 생성
# -------------------------
engine_no_db = create_engine(DB_URL_BASE, echo=True)

with engine_no_db.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    conn.commit()

print(f"✅ 데이터베이스 '{DB_NAME}' 준비 완료")

# -------------------------
# 2. DB 연결
# -------------------------
engine = create_engine(f"{DB_URL_BASE}/{DB_NAME}", echo=True)

# -------------------------
# 3. FK 체크 비활성화
# -------------------------
with engine.connect() as conn:
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

    conn.execute(text("DROP VIEW IF EXISTS ml_base_view"))

    conn.execute(text("DROP TABLE IF EXISTS parking_car"))
    conn.execute(text("DROP TABLE IF EXISTS traffic"))
    conn.execute(text("DROP TABLE IF EXISTS car_month"))
    conn.execute(text("DROP TABLE IF EXISTS population"))
    conn.execute(text("DROP TABLE IF EXISTS car"))
    conn.execute(text("DROP TABLE IF EXISTS cctv"))
    conn.execute(text("DROP TABLE IF EXISTS vehicle"))
    conn.execute(text("DROP TABLE IF EXISTS public_transit"))
    conn.execute(text("DROP TABLE IF EXISTS district"))

    conn.commit()


# -------------------------
# 4. CSV → 테이블 생성
# -------------------------
def load_csv_to_db(csv_path, table_name):
    df = pd.read_csv(csv_path)

    # district_id 타입 안전장치
    if "district_id" in df.columns:
        df["district_id"] = df["district_id"].astype("int64")

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )
    print(f"✅ {table_name} 테이블 생성 완료")

# -------------------------
# 5. 메인 실행
# -------------------------
if __name__ == "__main__":
    print("🚀 DB 초기화 시작")

    load_csv_to_db("data/district.csv", "district")
    load_csv_to_db("data/population.csv", "population")
    load_csv_to_db("data/car.csv", "car")
    load_csv_to_db("data/cctv.csv", "cctv")
    load_csv_to_db("data/car_month.csv", "car_month")
    load_csv_to_db("data/public_transit.csv", "public_transit")
    load_csv_to_db("data/parking_car.csv", "parking_car")
    load_csv_to_db("data/vehicle.csv", "vehicle")
    load_csv_to_db("data/traffic.csv", "traffic")

    # -------------------------
    # 6. district PK 생성
    # -------------------------
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE district
            ADD PRIMARY KEY (district_id)
        """))
        conn.commit()

    # -------------------------
    # 7. FK 생성 함수
    # -------------------------
    def add_fk(table_name, fk_name):
        with engine.connect() as conn:
            conn.execute(text(f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT {fk_name}
                FOREIGN KEY (district_id)
                REFERENCES district(district_id)
            """))
            conn.commit()
            print(f"🔗 {fk_name} 생성 완료")

    add_fk("population", "fk_population_district")
    add_fk("car", "fk_car_district")
    add_fk("vehicle", "fk_vehicle_district")
    add_fk("public_transit", "fk_pt_district")
    add_fk("cctv", "fk_cctv_district")

    # -------------------------
    # 8. View 생성 (FK 기반)
    # -------------------------
    with engine.connect() as conn:
        conn.execute(text("DROP VIEW IF EXISTS ml_base_view"))

        conn.execute(text("""
            CREATE VIEW ml_base_view AS
            SELECT
                d.district,
                p.district_id,
                p.datetime,
                p.population,
                p.population_diff,
                c.car_count,
                c.car_diff
            FROM population p
            JOIN car c
              ON p.district_id = c.district_id
             AND p.datetime = c.datetime
            JOIN district d
              ON p.district_id = d.district_id
        """))

        conn.commit()

    # -------------------------
    # 9. FK 체크 복구
    # -------------------------
    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        conn.commit()

    print("🎉 DB 초기화 + PK/FK + View 생성 완료")
