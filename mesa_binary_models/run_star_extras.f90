! ***********************************************************************
!
!   Copyright (C) 2012  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful, 
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************
 
      module run_star_extras 

      use star_lib
      use star_def
      use const_def
      use math_lib
      use binary_def
      use utils_lib, only: mesa_error
      
      implicit none
      
    contains

      subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s

         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         s% extras_startup => extras_startup
         s% extras_start_step => extras_start_step
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns

         s% other_accreting_state => other_accreting_state

      end subroutine extras_controls
      
      
      subroutine extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine extras_startup


      integer function extras_start_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_start_step = 0
      end function extras_start_step


      integer function extras_check_model(id)
         integer, intent(in) :: id
         extras_check_model = keep_going
      end function extras_check_model


      integer function how_many_extra_history_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 2
      end function how_many_extra_history_columns
      
      
      subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s

         integer :: i
         real(dp) :: moi, moi_const, mtot

         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         !moi = 0
         !mtot = 0
         !do i = s% nz, 1, -1
         !   moi = moi + (s% dm(i) * s% R(i) * s% R(i))
         !   mtot = mtot + s% dm(i)
         !end do

         moi = sum(s% dm(1:s% nz)*s% R(1:s% nz)*s% R(1:s% nz))
         mtot = sum(s% dm(1:s% nz))

         moi_const = moi / (mtot * s% R(1) * s% R(1))

         names(1) = "moment_of_intertia"
         vals(1) = moi  ! in SI units

         names(2) = "moment_of_intertia_constant"
         vals(2) = moi_const  ! in SI units

      end subroutine data_for_extra_history_columns

      
      integer function how_many_extra_profile_columns(id)
         integer, intent(in) :: id
         how_many_extra_profile_columns = 0
      end function how_many_extra_profile_columns
      
      
      subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
         integer, intent(in) :: id, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         integer :: k
         ierr = 0
      end subroutine data_for_extra_profile_columns
      

      integer function extras_finish_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_finish_step = keep_going

      end function extras_finish_step
      
      
      subroutine extras_after_evolve(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         real(dp) :: dt
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine extras_after_evolve

      subroutine other_accreting_state(id, total_specific_energy, accretion_pressure, accretion_density, ierr)
         use star_def
         integer, intent(in) :: id
         real(dp), intent(out) :: total_specific_energy, accretion_pressure, accretion_density
         real(dp) :: vesc_sq, total_specific_energy_surf, pressure_surf, density_surf, te, efactor
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         real(dp), dimension(:), allocatable :: change_in_dm, dm, prev_mesh_dm
         integer :: j

         ierr = 0
         ! before we can use controls associated with the star we need to
         ! get access 
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return 

         !allocate(prev_mesh_dm(s%nz), dm(s%nz), change_in_dm(s%nz))
         !call compute_prev_mesh_dm(s, prev_mesh_dm, dm, change_in_dm)

         !te = s% total_energy_profile_before_adjust_mass(1) / prev_mesh_dm(1)

         !total_specific_energy_surf = te
         pressure_surf = s%Peos(1)
         density_surf = s%rho(1)

         !print*,"ACC RATE DURING OTHER ACCSTATE CALL", s% mstar_dot

         efactor = 0.5
         vesc_sq = efactor * 2 * s% cgrav(1)*s% mstar/(s% photosphere_r*Rsun)
         total_specific_energy = (efactor - 1.0) * vesc_sq/(2 * efactor) ! erg/g
         !print*,"ACCRETION ENERGY: ", total_specific_energy
         
         accretion_density = s% mstar_dot/(4.0 * 3.141592653 * ((s% photosphere_r*Rsun)**2) * (vesc_sq**0.5)) ! g/cm^3
         !accretion_density = density_surf
         accretion_pressure = accretion_density * vesc_sq ! erg/cm^3

         accretion_density = MAX(density_surf, accretion_density)
         accretion_pressure = MAX(pressure_surf, accretion_pressure)

         !print*,"ACCRETION PRESSURE: ", accretion_pressure

      end subroutine other_accreting_state
      

      end module run_star_extras
      
