
#include <minizip/zip.h>
#include <minizip/unzip.h>
#include <fstream>
#include "zip.hpp"

#ifdef _WIN32
#define USEWIN32IOAPI
//#include "iowin32.h"
#endif

namespace yolo
{
	void log(const std::string_view& message);
}


namespace yolo::zip
{
	bool create_zip_file(const std::filesystem::path& dest_filename, const std::vector<std::filesystem::path>& files_to_zip)
	{
		std::string dest_filename_str = dest_filename.string();
		const char* dest_filename_cstr = dest_filename_str.c_str();

		zipFile zf = zipOpen(dest_filename_cstr, APPEND_STATUS_CREATE);
		if (zf == nullptr)
		{
			return false;
		}

		bool _return = true;
		for (size_t i = 0; i < files_to_zip.size(); i++)
		{
			const std::string path = files_to_zip[i].string();
			std::fstream file(path.c_str(), std::ios::binary | std::ios::in);
			if (file.is_open())
			{
				file.seekg(0, std::ios::end);
				long size = file.tellg();
				file.seekg(0, std::ios::beg);

				std::vector<char> buffer(size);
				if (size == 0 || file.read(&buffer[0], size))
				{
					zip_fileinfo zfi = {};
					auto fileName = files_to_zip[i].filename().string(); //path.substr(path.rfind('\\')+1);

					if (ZIP_OK == zipOpenNewFileInZip(zf, std::string(fileName.begin(), fileName.end()).c_str(), &zfi, nullptr, 0, nullptr, 0, nullptr, Z_DEFLATED, Z_NO_COMPRESSION))
					{
						if (zipWriteInFileInZip(zf, size == 0 ? "" : &buffer[0], size))
							_return = false;

						if (zipCloseFileInZip(zf))
							_return = false;

						file.close();
						continue;
					}
				}
				file.close();
			}
			_return = false;
		}

		if (zipClose(zf, nullptr))
		{
			return false;
		}

		if (!_return)
		{
			return false;
		}
		return true;
	}

#ifdef __APPLE__
	// In darwin and perhaps other BSD variants off_t is a 64 bit value, hence no need for specific 64 bit functions
#define FOPEN_FUNC(filename, mode) fopen(filename, mode)
#else
#define FOPEN_FUNC(filename, mode) fopen64(filename, mode)
#endif

	/// will overwrite files by default
	static int do_extract_currentfile(unzFile uf, const int* popt_extract_without_path, const char* password, const std::filesystem::path& dest_folder)
	{
		static const size_t WRITEBUFFERSIZE = 8192;

		char filename_inzip[256];
		char* filename_withoutpath;
		char* p;
		int err=UNZ_OK;
		FILE *fout=NULL;
		void* buf;
		uInt size_buf;

		unz_file_info64 file_info;
		err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip, sizeof(filename_inzip), NULL, 0, NULL, 0);

		if (err!=UNZ_OK)
		{
			printf("error %d with zipfile in unzGetCurrentFileInfo\n",err);
			return err;
		}

		size_buf = WRITEBUFFERSIZE;
		buf = (void*)malloc(size_buf);
		if (buf==nullptr)
		{
			printf("Error allocating memory\n");
			return UNZ_INTERNALERROR;
		}

		p = filename_withoutpath = filename_inzip;
		while ((*p) != '\0')
		{
			if (((*p)=='/') || ((*p)=='\\'))
			{
				filename_withoutpath = p+1;
			}
			p++;
		}

		if ((*filename_withoutpath)=='\0')
		{
			if ((*popt_extract_without_path)==0)
			{
				printf("creating directory: %s\n",filename_inzip);
				std::filesystem::create_directories(filename_inzip);
				//mymkdir(filename_inzip);
			}
		}
		else
		{
			std::string write_filename_str;

			if ((*popt_extract_without_path)==0)
				write_filename_str = filename_inzip;
			else
				write_filename_str = filename_withoutpath;
			write_filename_str = (dest_folder / write_filename_str).string();

			const char* write_filename = write_filename_str.c_str();

			err = unzOpenCurrentFilePassword(uf,password);
			if (err!=UNZ_OK)
			{
				printf("error %d with zipfile in unzOpenCurrentFilePassword\n",err);
			}

			if (err==UNZ_OK)
			{
				fout=FOPEN_FUNC(write_filename,"wb");
				/* some zipfile don't contain directory alone before file */
				if ((fout==NULL) && ((*popt_extract_without_path)==0) && (filename_withoutpath!=(char*)filename_inzip))
				{
					char c=*(filename_withoutpath-1);
					*(filename_withoutpath-1)='\0';
					std::filesystem::create_directories(write_filename);
					//makedir(write_filename);
					*(filename_withoutpath-1)=c;
					fout=FOPEN_FUNC(write_filename,"wb");
				}

				if (fout==NULL)
				{
					log("error opening '" + write_filename_str + "'");
				}
			}

			if (fout!=NULL)
			{
				//printf(" extracting: %s\n",write_filename);

				do
				{
					err = unzReadCurrentFile(uf,buf,size_buf);
					if (err<0)
					{
						log("error %d with zipfile in unzReadCurrentFile");
						break;
					}
					if (err>0)
					{
						if (fwrite(buf,(unsigned)err,1,fout)!=1)
						{
							log("error in writing extracted file");
							err=UNZ_ERRNO;
							break;
						}
					}
				}
				while (err>0);
				if (fout)
				{
					fclose(fout);
				}

				//if (err==0)
				//{
				//	change_file_date(write_filename,file_info.dosDate, file_info.tmu_date);
				//}
			}

			if (err==UNZ_OK)
			{
				err = unzCloseCurrentFile (uf);
				if (err!=UNZ_OK)
				{
					log("error %d with zipfile in unzCloseCurrentFile\n");
				}
			}
			else
			{
				unzCloseCurrentFile(uf); /* don't lose the error */
			}
		}

		free(buf);
		return err;
	}


	static int do_extract(unzFile uf,int opt_extract_without_path,const char* password, const std::filesystem::path& dest_folder)
	{
		uLong i;
		unz_global_info64 gi;
		int err;

		err = unzGetGlobalInfo64(uf,&gi);
		if (err!=UNZ_OK)
		{
			log("error '" + std::to_string(err) + "' with zipfile in unzGetGlobalInfo");
		}

		for (i=0;i<gi.number_entry;i++)
		{
			if (do_extract_currentfile(uf, &opt_extract_without_path, password, dest_folder) != UNZ_OK)
				break;

			if ((i+1)<gi.number_entry)
			{
				err = unzGoToNextFile(uf);
				if (err!=UNZ_OK)
				{
					log("error '" + std::to_string(err) + "' with zipfile in unzGoToNextFile");
					break;
				}
			}
		}

		return 0;
	}

	bool extract_zip_file(const std::filesystem::path& zip_filename, const std::filesystem::path& dest_folder)
	{
		std::string zip_filename_str = zip_filename.string();
		const char* zip_filename_cstr = zip_filename_str.c_str();
		unzFile uf = nullptr;
#ifdef USEWIN32IOAPI
		fill_win32_filefunc64A(&ffunc);
        uf = unzOpen2_64(zip_filename_cstr,&ffunc);
#else
		uf = unzOpen64(zip_filename_cstr);
#endif
		if (uf == nullptr)
		{
			log("Failed to open '" + zip_filename.string() + "' in 'extract_zip_file'");
			return false;
		}
		int ret = do_extract(uf, 0, nullptr, dest_folder);
		return ret == 0;
	}
}