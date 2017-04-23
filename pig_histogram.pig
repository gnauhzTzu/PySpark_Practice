register hdfs://user/tzu/myudfs.jar

raw = LOAD 'hdfs://user/tzu/2013-chunk-*' USING TextLoader as (line:chararray);

ntriples = foreach raw generate FLATTEN(myudfs.RDFSplit3(line)) as (subject:chararray,predicate:chararray,object:chararray);

-- group the n-triples by object column
subjects = group ntriples by (subject) PARALLEL 50;

count_by_subject = foreach subjects generate flatten($0), COUNT($1) as count PARALLEL 50;

x_axis = group count_by_subject by (count) PARALLEL 50;

y_axis = foreach x_axis generate flatten($0), COUNT($1) as final_count PARALLEL 50;

y_axis_ordered = order y_axis by final_count PARALLEL 50;

store y_axis_ordered into 'hdfs://user/tzu/output-7' using PigStorage();