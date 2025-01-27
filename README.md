# Overview

This repository contains everything you need to download and visualize the data
collected under the 1990s Clean Air Act. The US Environmental Protection Agency
maintains and makes public this data [here.](https://campd.epa.gov/data) They have an
[API](https://github.com/USEPA/cam-api-examples) that makes this data available through
a website GUI and also through scripts, and it's where I started when writing the
downloading script that is used here.

The purpose of this repository is to show people things I learned about working with
medium to large datasets. In particular, I'll show how to obtain the raw data, create a
parquet dataset, query it with SQL using `duckdb`, and build a visualization dashboard
using Plotly Dash.

As a disclaimer, I'll mention that while I work at the EPA in a group that deals
directly with this data, everything I write here is as a public citizen and does not
reflect the EPA in any way.

# Getting the Data

The data is made public in uncompressed `csv` files. This is a poor choice, as the
entirety of the data totals to about 200GB, when compressed, it would be only 9GB.
Therefore, the scipt located in

    campd_visualizer/scripts/get_campd_data.py

downloads the quarterly data as `csv` files and immediately compresses them into zip
files, so that you will never need 200GB of free space on your hard drive. This will
take about 8 hours, and on Windows, it kept crashing so I had to keep rerunning it.

The data about the power generation facilities is also made available through yearly csv
files, which are downloaded in the same way.

# Preparing the Data for Analysis and Visualization

After the data is downloaded, there is a question about how it should be accessed.
Perhaps most simply, one could read the files in the `pandas` library, as it can even
read compressed files. But on reflection this is a poor choice. `pandas` typically loads
the data into memory, which is not practical for large datasets. Further, with the data
spread out over so many quarterly csvs, it would be a pain to figure out which files are
relevant.

An SQL database would offer some advantages here. By loading all of the quarterly data
into a single SQL table, we could conveniently query it to get whatever data we wanted.
And because such databases are memory-efficient, we wouldn't be loading things into RAM
unnecessarily.

However, such databases are annoying to install and setup. Furthermore, as far as my
experience goes, when I built an `sqlite` database of this data, the database itself was
close to 200GB in size, the same as the uncompressed `csv` files. Furthermore, `sqlite`
is somewhat poorly setup to handle data as large as this, missing features to import
data into a database table, and lacking the option to perform indexing on the table. SQL
indices serve as metadata that is used so that all of the data doesn't need to be looked
at to return query results. While `postgresql` has some more possibilities for
optimization, the database was still 200GB in size. Not great.

SQL databases are also in some sense overkill, because for data analysis tasks we are
not typically going to ever change any of the data. So all of the infrastructure that
SQL databases have to facilitate that are of no use.

A popular modern approach to this situation is to use parquet files. These are basically
meant for read-only applications, and this might be what allows them to compress the
data. In particular, the entire dataset is brought down to about 5.6GB. It achieves this
by storing a table not as a series of rows, but as columns, which allows for more
straightforward compression. It also keeps metadata in each of it's files, referred to
as row statistics, which help the parquet reader to know whether a query needs any of
the data in the file or not. This is analogous to an SQL index, but parquet does this
automatically. Still, when you know that queries will occur along certain dimensions,
say Year or State, you can manually specify these columns as a partition column, and the
parquet dataformat will break your data which spans over N years into N files in N
subdirectories. The most common implementation of this is called Hive Partitioning.

As I've alluded to, you can actually issue standard SQL queries to a parquet dataset
through a popular tool called `duckdb`. And so for the user, there is very little
difference getting data from an SQL database or from a parquet dataset. There seems to
be no performance loss using `duckdb` and parquet files either. And all this, without
having to deal with oversized files. I believe this is a winner, and so I will show how
to make this dataset.

After downloading the data as mentioned above, run

    campd_visualizer/scripts/zip2parquet.py

This creates a parquet dataset, with the data partitioned by year. How long it takes
depends on your hardware, so maybe 25min to 3hrs. But the parquet dataset is itself not
really a unique end point. What I mean is that although the data has been partitioned by
year into different folders, if you look in them, you'll see a horrendously large number
of files. A parquet reader would now need to look at each of these files, examine the
metadata, and decide whether or not it needs to be read, and if so, where. So there is a
cost to having files that are too small. On the other hand, there is a cost to having
the data in too few files, because in that case a larger file must be opened up and
examined, and if only a little bit of data is needed, there can be a lot of wasted
effort.

So a balance is needed. I read somewhere that the individual files should probably be
between 20MB and 2GB. When this dataset is partitioned by Year alone, the directory
contains about 200MB of data, which can comfortably fit in a single file. If we
paritioned by Year and by state, this would create about 1500 files, each about 4MB. For
this reason, I decided to partition only on Year.

Next, to address the fact that all of these tiny files were created, the common solution
is to repartition the dataset. Meaning, once the dataset has been created with
unoptimized file sizes, it can then be remade, using this as a starting point, to
recreate files. This is accomplished by

    campd_visualizer/scripts/repartition_master.py

How long it takes depends on your hardware, so maybe 25min to 3hrs. In this file, you
may need to set the `PARTITION_SIZE` parameter at the bottom. This is a hardware
dependent parameter, and I believe it reflects how much RAM the parquet writer can use
when creating a file. If it hits this value, then it will write what it has to file and
start another. So your hardware will limit the largest file you can write. To get all of
the data for each year into a single file, I think you need about 30GB of RAM. I have
128GB, so I didn't really think too much about it. When building on a laptop with only
16GB of RAM, I had to set this parameter to 5000MB to not max out the system RAM, and
this generates about 4 files per year.

### Required Software

You'll need `python` installed, and `conda` to create the environment I use. On Linux,
you need at least 8GB of RAM, I'd up that to 16GB for Windows. I primarily use Linux, so
I know all this works pretty well there. I tested this in Windows, but not all that
much. If you use Windows, maybe don't.

### `scripts/get_camd_data.py`

This script downloads all of the hourly emmissions data since the 1990s when the clean
air act started this data collection. To run it, you will need to apply for a key
[here.](https://www.epa.gov/airmarkets/cam-api-portal#/api-key-signup) It should only
take minutes to get the key in your email. Place whatever that string is into a file at
`keys/camd_key`.

You will also have to install this python package as well so that python knows where to
look to complete the import statements. You can do this by navigating to the root
directory, the one with `setup.py`, and running

    pip install -e .

I recommend you run this command within some python environment of your choice.

You can then view the few parameters in the download script. I don't think you should
need to change anything. Simply run it with `python get_camd_data.py`. It's about 200GB
of data, and I think it took me about 8 hours to download.
