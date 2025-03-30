Topic Model documentation

Regular Topic Model



Segment Topic Model

To create a segment for building a topic model. 

1. Create a helper segment in make_segment_table.sql. 
2. Use helper segment to add data to the actual segment. This will have an empty mongo_tokens column
3. Make a new mongo collection for that segment's tokens. Don't forget to give it an index.
4. Use fetch_segment_tokens.py as 'Fetch keywords list and make tokens' to collect the keywords, and turn them into tokens. You will have to update the collection info. Remember that the first 30k are borked and have no keys or descriptions so you have to have >30K items in your LIMIT. This will set mongo_tokens = 1. You may/will need to temporarily change the SegmentBig_isnotface tablename in my_declarativebase.py file.
5. Create a ImagesTopics_segmentname table to hold the topic assignments. 
6. Use topic_model.py with 'Index topics' to assign a topic to each image. Be careful which model you use and how you select, which is controlled by IS_NOT_FACE and USE_EXISTING_MODEL