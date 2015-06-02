!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief  Allocates the data for each mesh chunk
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The data fields for the mesh chunk are allocated based on the mesh
!>  size.

SUBROUTINE build_field(chunk,x_cells,y_cells,z_cells)

   USE tea_module

   IMPLICIT NONE

   INTEGER :: chunk,x_cells,y_cells,z_cells

   chunks(chunk)%field%x_min=1
   chunks(chunk)%field%y_min=1
   chunks(chunk)%field%z_min=1

   chunks(chunk)%field%x_max=x_cells
   chunks(chunk)%field%y_max=y_cells
   chunks(chunk)%field%z_max=z_cells

   IF (use_opencl_kernels .EQV. .TRUE.) THEN
     call initialise_ocl(chunks(chunk)%field%x_min, &
                          chunks(chunk)%field%x_max, &
                          chunks(chunk)%field%y_min, &
                          chunks(chunk)%field%y_max, &
                          chunks(chunk)%field%z_min, &
                          chunks(chunk)%field%z_max)
  ENDIF
  
END SUBROUTINE build_field
