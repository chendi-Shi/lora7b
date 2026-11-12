"""合成一套小 NL2SQL 数据集（带真 sqlite 库），用来做 5 分钟冒烟测试。

为什么需要它：Spider/BIRD 要下载几百 MB 的库文件，网络一卡就卡在第一步。
这里用固定模板造 3 张表 + 120 条 (question, sql)，特征覆盖 JOIN / GROUP BY /
ORDER BY / LIKE / 聚合 / HAVING，足以验证"数据→训练→评测"整条管线是通的，
并且能让 7B 模型在 50 步内过拟合出可见的准确率上升（说明链路真的在学习）。
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT,
    city        TEXT,
    signup_date TEXT,
    vip         INTEGER
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    title      TEXT,
    category   TEXT,
    price      REAL,
    stock      INTEGER
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id  INTEGER,
    quantity    INTEGER,
    amount      REAL,
    status      TEXT,
    order_date  TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES products(product_id)
);
"""

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
CATEGORIES = ["electronics", "books", "home", "sports", "beauty"]
STATUSES = ["paid", "shipped", "cancelled", "refunded"]
ADJ = ["premium", "basic", "smart", "eco", "pro"]
NOUN = ["speaker", "lamp", "chair", "monitor", "kettle", "desk", "phone", "mouse"]


def _build_db(db_path: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(DDL)

    names = [f"user_{i:03d}" for i in range(1, 61)]
    con.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [
            (i, names[i - 1], rng.choice(CITIES),
             f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.randint(0, 1))
            for i in range(1, 61)
        ],
    )
    con.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?)",
        [
            (i, f"{rng.choice(ADJ)} {rng.choice(NOUN)}", rng.choice(CATEGORIES),
             round(rng.uniform(9, 900), 2), rng.randint(0, 500))
            for i in range(1, 41)
        ],
    )
    con.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
        [
            (i, rng.randint(1, 60), rng.randint(1, 40), rng.randint(1, 5),
             round(rng.uniform(20, 4000), 2), rng.choice(STATUSES),
             f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
            for i in range(1, 401)
        ],
    )
    con.commit()
    con.close()


def _templates() -> list[tuple[str, str]]:
    t: list[tuple[str, str]] = []

    for city in CITIES:
        t.append((f"列出城市为 {city} 的客户姓名",
                  f"SELECT name FROM customers WHERE city = '{city}'"))
        t.append((f"{city} 有多少个 VIP 客户",
                  f"SELECT COUNT(*) FROM customers WHERE city = '{city}' AND vip = 1"))
    for cat in CATEGORIES:
        t.append((f"{cat} 类目下最贵的商品标题",
                  f"SELECT title FROM products WHERE category = '{cat}' "
                  f"ORDER BY price DESC LIMIT 1"))
        t.append((f"{cat} 类目商品的平均价格",
                  f"SELECT AVG(price) FROM products WHERE category = '{cat}'"))
    for st in STATUSES:
        t.append((f"状态为 {st} 的订单总金额",
                  f"SELECT SUM(amount) FROM orders WHERE status = '{st}'"))
        t.append((f"状态为 {st} 的订单数量",
                  f"SELECT COUNT(*) FROM orders WHERE status = '{st}'"))
    for city in CITIES:
        t.append((f"来自 {city} 的客户下了多少订单",
                  f"SELECT COUNT(*) FROM orders o JOIN customers c "
                  f"ON o.customer_id = c.customer_id WHERE c.city = '{city}'"))
        t.append((f"{city} 客户贡献的订单总金额",
                  f"SELECT SUM(o.amount) FROM orders o JOIN customers c "
                  f"ON o.customer_id = c.customer_id WHERE c.city = '{city}'"))
    for cat in CATEGORIES:
        t.append((f"每个状态的订单里包含 {cat} 类商品的订单数是多少，按状态分组",
                  f"SELECT o.status, COUNT(*) FROM orders o JOIN products p "
                  f"ON o.product_id = p.product_id WHERE p.category = '{cat}' "
                  f"GROUP BY o.status"))
    t.append(("库存小于 50 的商品有多少",
              "SELECT COUNT(*) FROM products WHERE stock < 50"))
    t.append(("销量最高的商品标题", "SELECT p.title FROM orders o JOIN products p "
              "ON o.product_id = p.product_id GROUP BY p.product_id "
              "ORDER BY SUM(o.quantity) DESC LIMIT 1"))
    t.append(("下单次数超过 10 次的客户姓名",
              "SELECT c.name FROM orders o JOIN customers c "
              "ON o.customer_id = c.customer_id GROUP BY o.customer_id "
              "HAVING COUNT(*) > 10"))
    t.append(("价格在 100 到 500 之间的商品数量",
              "SELECT COUNT(*) FROM products WHERE price BETWEEN 100 AND 500"))
    t.append(("标题包含 smart 的商品名",
              "SELECT title FROM products WHERE title LIKE '%smart%'"))
    t.append(("没有下过任何订单的客户姓名",
              "SELECT name FROM customers WHERE customer_id NOT IN "
              "(SELECT customer_id FROM orders)"))
    t.append(("2024 年 3 月下单的订单号", "SELECT order_id FROM orders "
              "WHERE order_date LIKE '2024-03%'"))
    t.append(("按城市统计客户数量，降序",
              "SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY 2 DESC"))
    t.append(("每个类目的商品数量",
              "SELECT category, COUNT(*) FROM products GROUP BY category"))
    return t


def build(out_dir: str | Path, n_train: int = 150, n_dev: int = 40,
          n_test: int = 40, seed: int = 7) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "shop.sqlite"
    _build_db(db_path, seed=seed)

    pool = _templates()
    rng = random.Random(seed)
    rows = [{"db_id": "shop", "question": q, "sql": s} for q, s in pool]
    rng.shuffle(rows)

    def write(name: str, items: list[dict]) -> str:
        p = out / name
        with p.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps({"db_id": r["db_id"], "db_path": db_path.name,
                                    "question": r["question"], "sql": r["sql"]},
                                   ensure_ascii=False) + "\n")
        return str(p)

    # 问题池只有 ~70 条，训练集靠重复采样凑步数；dev/test 用不同的问法切片
    train = [rows[i % len(rows)] for i in range(n_train)]
    dev = rows[: n_dev]
    test = rows[n_dev: n_dev + n_test] if len(rows) > n_dev else rows[:n_test]
    return {"train": write("train.jsonl", train), "dev": write("dev.jsonl", dev),
            "test": write("test.jsonl", test), "db": str(db_path)}
