cd /scratch/map_data
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld --max-trip-size 10000 --max-viaroute-size 10000 --max-table-size 10000 /data/map.osrm