import copy
import json
import os
from pathlib import Path

import subprocess

from loguru import logger
from getpass import getpass
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.splitters import RollingWindowSplitter
#from semantic_router.utils.logger import logger


encoder = HuggingFaceEncoder(
    name="BAAI/bge-m3",
    device="cpu" #cuda 
)

# the following lines are meant to be used with GPU

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


splitter_pdf = RollingWindowSplitter(
    encoder=encoder,
    dynamic_threshold=True,
    min_split_tokens=150,
    max_split_tokens=300,
    window_size=5,
    plot_splits=False,  # set this to true to visualize chunking
    enable_statistics=False  # to print chunking stats
)

splitter_txt = RollingWindowSplitter(
    encoder=encoder,
    dynamic_threshold=True,
    min_split_tokens=100,
    max_split_tokens=200,
    window_size=2,
    plot_splits=False,  # set this to true to visualize chunking
    enable_statistics=False  # to print chunking stats
)

#to be used with GPU


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=False,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=False,  # Whether to dump middle JSON files
    f_dump_model_output=False,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=False,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)
        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)
            pdf_info = middle_json["pdf_info"]
            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )
            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )
                return md_content_str
            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )
            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )
            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )
            logger.info(f"local output dir is {local_md_dir}")
    else:
        print("Backend not found")

        
def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="en",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse_result = do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
        return do_parse_result
    except Exception as e:
        logger.exception(e)
        
        
def get_pdf_chunks(doc_path_list, output_dir, lenguage="en"):
    parsed = parse_doc([doc_path_list], output_dir, lang=lenguage, backend="pipeline")
    splits = splitter_pdf([parsed])
    chunks = []
    for i in range(len(splits)):
        chunks.append(splits[i].docs)
    return chunks


def get_txt_chunks(txt):
    splits = splitter_txt([txt])
    chunks = []
    for i in range(len(splits)):
        chunks.append(splits[i].docs)

    return chunks



def get_doc_chunks(doc_path, output_dir, lenguage="en"):
    #"en"
    
    os.makedirs(output_dir, exist_ok=True)

    # Convert DOCX to PDF using LibreOffice
    pdf_filename = os.path.splitext(os.path.basename(doc_path))[0] + ".pdf"
    output_pdf_path = os.path.join(output_dir, pdf_filename)

    subprocess.run([
        "soffice", "--headless", "--convert-to", "pdf",
        "--outdir", output_dir, doc_path
    ], check=True)
    
    parsed = parse_doc([output_pdf_path], output_dir, lang=lenguage, backend="pipeline")
    splits = splitter_pdf([parsed])
    
    chunks = []
    for i in range(len(splits)):
        chunks.append(splits[i].docs)
    return chunks
