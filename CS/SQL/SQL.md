## SQL Learning

==**https://dev.mysql.com/doc/refman/8.0/en/**==

[toc]

#### Command

##### Login

> - ` mysql -u root -p`

##### Database

> - `SHOW DATABASES;`
> - `CREATE DATABASE test;`
> - `DROP DATABASE test;`

##### Table

> - `USE test;`
>
> - `SHOW TABLES:`
>
> - ```mysql
>   DESC <table>;
>   ```
>
>     : 查看表的结构
>
> - `SHOW CREATE TABLE students;`
>
> - `CREATE TABLE <table>;`
>
> - `DROP TABLE <table>;`
>
> - `ALTER TABLE students ADD COLUMN birth VARCHAR(10) NOT NULL;`  ：增加列
>
> - `ALTER TABLE students CHANGE COLUMN birth birthday VARCHAR(20) NOT NULL;` ：修改列
>
> - `ALTER TABLE students DROP COLUMN birthday;` : 删除列
>
> - `CREATE TABLE pet(name VARCHAR(20), owner VARCHAR(20));`

#### Load data

> 可以创建txt文件，以tab分隔，进行文件导入 NULL值可以用 **\N** 来表示
>
> ```mysql
> LOAD DATA LOCAL INFILE '/path/pet.txt' INTO TABLE pet;
> ```
>
> ```mysql
> LINES TERMINATED BY '\r'
> ```
>
> ```mysql
> INSERT INTO pet VALUES ('', '', '', NULL, );
> ```
>
> 



#### Query

`SELECT * FROM table`

##### Condition query  [WHERE]

> `SELECT * FROM students WHERE score < 80 AND gender ='M';`
>
> `SELECT * FROM students WHERE score BETWEEN 60 AND 90; `

> `SELECT * FROM students WHERE score < 80 OR gender ='M';`

> `SELECT * FROM students WHERE NOT class_id = 2;`

> not equal : `score <> 80`

> Similar: `name LIKE 'ab%'`

**Priority:** `NOT` > `AND` > `OR`



##### Hibernate query

> `SELECT id, score, name FROM students`

> **new column name** 
>
> `SELECT id, score, name newname FROM students`

> `SELECT score points, name FROM students WHERE gender = 'M';`



##### Sort

> `SELECT * FROM students ORDER BY score;` [升序] (ASC)
>
> `SELECT * FROM students ORDER BY score DESC; ` [降序]

> `SELECT * FROM students ORDER BY score DESC, gender;`

> `SELECT id, name, gender, score
> FROM students
> WHERE class_id = 1
> ORDER BY score DESC;`

##### 数据计算

> ``` mysql
> SELECT name, birth, CURDATE(), TIMESTAMPDIFF(YEAR, birth, CURDATE()) AS age from pet;
> ```

##### 模式匹配

> LIKE,  NOT LIKE
>
> ```mysql
> select * from students where name like '小%';
> ```
>
> _ 匹配单个字符， % 匹配任意数字的字符；
>
> ```mysql
> REGEXP_LIKE() 进行匹配. fy结尾'fy$', b开头'^b', 含有w'w'
> '^.....$' / '^.{5}$'
> ```

##### 记录行数

> ```mysql
> SELECT COUNT(*) from pet;
> ```
>
> 

##### 分页查询

> `SELECT id, name, gender, score
> FROM students
> ORDER BY score DESC
> LIMIT 3 OFFSET 3;`
>
> : LIMIT 单页记录限制， OFFSET起始记录索引

##### 聚合查询

> `SELECT COUNT(*) FROM students;`
>
> `SELECT COUNT(*) num FROM students;`
>
> `SELECT COUNT(*) boys FROM students WHERE gender = 'M';`

> **SUM, AVG, MAX, MIN**
>
> `SELECT AVG(score) average FROM students WHERE gender = 'M';`

##### 分组聚合

> `SELECT class_id, COUNT(*) num FROM students GROUP BY class_id;`

##### 多表查询

> `SELECT * FROM students, classes;`
>
> `SELECT
>     students.id sid,
>     students.name,
>     students.gender,
>     students.score,
>     classes.id cid,
>     classes.name cname
> FROM students, classes;`
>
> **TABLE.COLUMN** 

##### 连接查询

> `SELECT s.id, s.name, s.class_id, c.name class_name, s.gender, s.score
> FROM students s
> INNER JOIN classes c
> ON s.class_id = c.id;`
>
> **INNER JOIN** 表 **ON** 连接条件 ： 返回同时存在于两张表的行数据
>
> **RIGHT OUTER JOIN** ： 返回有表存在的数据
>
> **LEFT OUTER JOIN** ： 返回做表存在的数据
>
> **FULL OUTER JOIN**



#### Insert

> `INSERT INTO <table> (column1, ...) VALUES (value1, ...)`
>
> `INSERT INTO students (class_id, name, gender, score) VALUES (2, '大牛', 'M', 80);`
>
> `INSERT INTO students (class_id, name, gender, score) VALUES
>   (1, '大宝', 'M', 87),
>   (2, '二宝', 'M', 81);`

#### Update

> `UPDATE <table> SET column_name1 = value1 ,... WHERE ...;`
>
> `UPDATE students SET name='大牛', score=66 WHERE id=1;`

> `UPDATE students SET score=score+10 WHERE score<80;`

#### Delete

> `DELETE FROM <table> WHERE ...;`



#### Useful

> `REPLACE INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99);` **==有则修改，无则增加==**

> `INSERT INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99) ON DUPLICATE KEY UPDATE name='小明', gender='F', score=99;` **==插入或更新==**

> `INSERT IGNORE INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99); ` **==插入或忽略==**



> `CREATE TABLE students_of_class1 SELECT * FROM students WHERE class_id=1;`  ==**复制到新表**==

> FORCE_INDEX 强制使用指定索引



### 事务

> - 显示事务
>
> `BEGIN; `
>
> `...`
>
> `COMMIT;`

> `ROLLBACK;` 回滚事务，整个事务将失败



### 隔离级别

> -  Read Uncommitted
> - Read Committed
> - Repeatable Read
> - Serializable



```mysql
CREATE TABLE pet (name VARCHAR(20), owner VARCHAR(20), species VARCHAR(20), sex CHAR(1), birth DATE, death DATE);


```



### 函数

https://dev.mysql.com/doc/refman/8.0/en/sql-function-reference.html

Batch 模式

`mysql < batch-file`



定义变量

select @variable= ;

