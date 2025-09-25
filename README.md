# Supermarket_Agent

功能介绍：
1.挂载RAG数据库的超市AIAgent智能客服，实现与客户的智能交互，方便客户询问商品信息（折扣、库存、品牌、位置、保质期等），理解用户的模糊需求并做相关推荐例如：“我想为新装修好的厨房增添厨具”，AIAgent能给顾客推荐相关产品，列举出商品的详细信息，并可以反复对话帮助用户确定所需要的商品。支持csv和PDF的数据导入。
2.支持端到端生成并执行python代码进行数据分析的Agent。超市管理员可传入商品信息，提出需求，可以实现自动数据分析、生成数据图表。支持csv数据导入
3.支持向量数据库的管理，上传的数据会进行哈希计算查看是否已存在，如果是新传入的数据，则会永久保存文件和embedding后的向量数据库。支持预览、删除、切换多个文件的向量数据库。

项目开始：
第一步：pip install -r requirements.txt
第二步：获取LLM和embedding的API，填写在.env文件下的“DEEPSEEK_API_KEY=”和“DASHSCOPE_API_KEY=”的后面
本项目使用的是deepseek的LLM服务和阿里的dashscope的embedding服务
获取deepseek API：https://platform.deepseek.com/usage
获取Dashscope API:https://bailian.console.aliyun.com/?tab=model#/api-key
第三步：
在python环境所在的终端运行streamlit run supermarket_agent.py

页面展示：
采用streamlit实现前端交互
页面引用photo文件夹内图片

代码版本迭代：
v1：基础功能实现
v1.1：修复点击快速询问按钮后，无法实现询问-回复的效果
v1.1.1： 特别版本，实现本地embedding模型（后续仍采用DashScope云embedding模型）
v1.2 实现向量数据库的永久存储和管理，利用哈希编码判断是否需要进行向量化，减少重复数据的向量化需求
v1.3 优化向量数据库的文件管理，所有数据存储在data_backup文件夹下，上传的csv存储在data_backup/saved_files。向量数据库存储在data_backup/db下，存储的数据以时间戳分别命名

实验数据说明：
原始实验数据下载地址：https://gitee.com/EricLiuCN/barcode

预处理后数据,历代数据更新存储在：data目录下
data/barcodes_v1:去除不必要列，清洗brand列为空的值，数据量14135，9列

data/barcodes_v2:测试数据，随机抽取barcodes_v1中的数据，数据量1414，9列

data/barcodes_v3_admission.csv：在data/barcodes_v1基础上进行更新
1.清洗brand列不正常的值，例如"-"，
2.将price列由object类别更新为float类型，加快Agent查询效率
3.新增列discount,inventory,expiration_data,product_location
均为随机生成，用于测似，其中
    discount：有10%商品为0.9，5%为0.8，1%为0.7，其余均为1
    inventory: 范围20-50
    expiration: 范围范围2026年9月1日-2027年9月30日
    product_location: 在ABCDEFGH这8个区
数据量14065行，13列
！！！此数据文件包含信息更多，推荐用于超市管理员的Agent数据分析功能

data/barcodes_v3_customer.csv：在data/barcodes_v3_admission.csv基础上进行更新
去除barcode,supplier,madein三种列，对于超市顾客来说此类信息几乎并不影响体验，极大提升Agent的运行效率，主要用于搭建面向超市顾客地RAG数据库

co-worker:
1.余子轩






