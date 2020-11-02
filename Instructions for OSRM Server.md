This is kinda confusing in parts, so I thought I'd put some clearer instructions up, mostly so I can remember the exact commands

1. Get Docker running
   
The easiest way to use the OSRM server is through docker. I have no idea how to do this on mac or windows, although it doesn't look to hard.

The only trick I've found is that on linux, docker doesn't start automatically. It can be started using `systemctl start docker`.

2. Download the server

This is really easy, although I have no idea where the download location is.

In the command prompt (or console/whatever), run `docker pull osrm/osrm-backend`. That's it.

3. Download map data

This is somewhat trickier. The file formats *.osm, *.osm.pbf and *.osm.bz2 are supported.

The way I've done it is by going to https://www.openstreetmap.org/export#map=12/-43.5258/172.6730 and downloading an area.
Note that if the area is too large, this can fail. I just tried the mirrors listed until one worked.

For the experiments we've run, the map file used is hosted at https://drive.google.com/file/d/1gVyb4wfWogdUbIhFy5iZLECEc5f92mU4/view?usp=sharing

4. Clean map data

Navigate to the folder the map data is downloaded to. Run the following commands, where "filename.osm" is replaced by the name of the map data file.
Note that the command maps the current directory to the /data directory in the container, so /data/filename.osm works while /filename.osm will not.

Also note that after the first command, the new file filename.osrm is created, which should be input in the next commands.

```
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/filename.osm
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/filename.osrm
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/filename.osrm
``` 

5. Start the server.

All the above steps only need to be done once. After that, to start the server, run:

```
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/filename.osrm
```
for larger docker limit
```
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld --max-trip-size 10000 --max-viaroute-size 10000 --max-table-size 10000 /data/map.osrm
```

Again, in the folder with the map data and the correct filename.orsm.

By default, this will host the server on `localhost:5000`.