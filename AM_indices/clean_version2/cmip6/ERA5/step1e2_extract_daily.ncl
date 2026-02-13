;zg

dir  = "/data/A/reanalyses/ERA-5/"

year = ispan(1950,2021,1)
nyr  = dimsizes(year)

fin = addfile(dir+"1950/ERA5.pl.1950-01-01.nc","r")
time0 = fin->time
lat = fin->latitude
lon = fin->longitude
lev = fin->level
nlat=dimsizes(lat)
nlon=dimsizes(lon)
nlev=dimsizes(lev)

NAM = new((/nyr*365,nlev/),float)
SAM = new((/nyr*365,nlev/),float)
Global = new((/nyr*365,nlev/),float)
time = new((/nyr*365/),integer)
copy_VarAtts(time0, time)


do iy = 0,nyr-1
    
    zg = new((/365,nlev,nlat,nlon/),float)

    files := systemfunc("ls "+dir+year(iy)+"/ERA5.pl."+year(iy)+"*")
    ndays = dimsizes(files)
    iday = 0
    if ndays .eq. 366 then
        do ifil = 0,366-1
            if ifil .eq. 59 then ; remove 2.29
                print("   "+files(ifil))
                continue
            end if
            print(""+files(ifil))
            fin = addfile(files(ifil),"r")
            zg(iday,:,:,:) = (/ dim_avg_n(short2flt(fin->z(:,:,:,:)),0) /)
            time(iy*365+iday) = (/ fin->time(0) /)
            iday = iday+1
        end do
    else
        do ifil = 0,365-1
            print(""+files(ifil))
            fin = addfile(files(ifil),"r")
            zg(iday,:,:,:) = (/ dim_avg_n(short2flt(fin->z(:,:,:,:)),0) /)
            time(iy*365+iday) = (/ fin->time(0) /)
            iday = iday+1
        end do
    end if

; Geopotential to geopotential height
zg = zg/9.8

zg!2="lat"
zg&lat=lat
zg!3="lon"
zg&lon=lon

rad    = 4.0*atan(1.0)/180.0
clat   = cos(lat*rad)
copy_VarCoords(lat,clat)
NAM(iy*365:iy*365+364,:) = wgt_areaave(zg(:,:,{65:90},:),clat({65:90}),1.0,0)
SAM(iy*365:iy*365+364,:) = wgt_areaave(zg(:,:,{-90:-65},:),clat({-90:-65}),1.0,0)
Global(iy*365:iy*365+364,:) = wgt_areaave(zg,clat,1.0,0)

delete([/zg,rad,clat/])

end do

NAM!0="time"
NAM&time=time
NAM!1="level"
NAM&level=lev
copy_VarCoords(NAM, SAM)
copy_VarCoords(NAM, Global)


;save output
setfileoption("nc","Format","NetCDF4")
system("rm -r AM_daily_1950_2021.nc")
fout = addfile("AM_daily_1950_2021.nc","c")
fout->NAM=NAM
fout->SAM=SAM
fout->Global=Global


