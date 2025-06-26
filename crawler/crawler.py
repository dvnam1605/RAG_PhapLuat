import scrapy
from bs4 import BeautifulSoup
from scrapy_playwright.page import PageMethod
import re
import logging
import os
import hashlib

# Thiết lập logger
spider_logger = logging.getLogger(__name__)

class TvplSpiderFinal(scrapy.Spider):
    """
    Spider để cào dữ liệu văn bản pháp luật từ thuvienphapluat.vn.
    - Sử dụng Playwright để xử lý các trang có JavaScript.
    - Tự động phân trang qua các trang kết quả tìm kiếm.
    - Trích xuất nội dung chi tiết của từng văn bản.
    - Có thể chạy độc lập bằng lệnh: scrapy runspider play.py
    """
    name = "tvpl_spider"
    allowed_domains = ["thuvienphapluat.vn"]

    custom_settings = {
        # Cấu hình cốt lõi để Playwright hoạt động
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        # Giả mạo User-Agent để tránh bị chặn (lỗi 403 Forbidden)
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',

        # Cài đặt tối ưu và kiểm soát hành vi crawl
        'ROBOTSTXT_OBEY': False, # Bỏ qua robots.txt để tránh bị chặn
        'DOWNLOAD_DELAY': 1,     # Độ trễ giữa các request để giảm tải cho server
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4, # Giới hạn số request đồng thời để ổn định hơn
        'RETRY_TIMES': 3,        # Thử lại 3 lần nếu gặp lỗi
        'LOG_LEVEL': 'INFO',     # Mức log, đổi thành 'DEBUG' để xem chi tiết khi gỡ lỗi

        ## FIX 1: Tăng thời gian chờ điều hướng mặc định từ 30s lên 60s.
        ## Điều này giải quyết trực tiếp lỗi TimeoutError cho các trang tải chậm.
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000, # 60,000 milliseconds = 60 seconds
        

        # Cấu hình Playwright
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True  # Chạy ở chế độ ẩn, đổi thành False để xem trình duyệt hoạt động
        },
        # Tự động hủy các request không cần thiết (ảnh, css) để tăng tốc
        'PLAYWRIGHT_ABORT_REQUEST': lambda req: req.resource_type in ("image", "stylesheet", "font"),
    }

    # URL khởi đầu, có thể thay đổi trang bắt đầu ở đây
    start_urls = [
        "https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=1&match=True&type=0&status=0&signer=&sort=1&lan=0&scan=0&org=0&fields=&page=120"
    ]

    # Giới hạn số trang kết quả tìm kiếm sẽ crawl, 0 là không giới hạn
    MAX_SEARCH_RESULT_PAGES = 1000
    search_pages_crawled_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spider_logger.critical("<<<<< TVPL_SPIDER: __INIT__ CALLED >>>>>")
        # Cho phép ghi đè số trang tối đa từ dòng lệnh (ví dụ: -a max_pages=10)
        if hasattr(self, 'max_pages'):
            try:
                self.MAX_SEARCH_RESULT_PAGES = int(self.max_pages)
                spider_logger.info(f"Overriding MAX_SEARCH_RESULT_PAGES with cmd arg: {self.MAX_SEARCH_RESULT_PAGES}")
            except ValueError:
                spider_logger.warning(f"Invalid value for -a max_pages: '{self.max_pages}'. Using default: {self.MAX_SEARCH_RESULT_PAGES}")

        # Tạo thư mục output nếu chưa có
        self.output_dir = 'output_texts'
        os.makedirs(self.output_dir, exist_ok=True)
        spider_logger.info(f"Output directory set to: '{self.output_dir}'")

    def sanitize_filename(self, filename):
        """Làm sạch chuỗi để sử dụng làm tên file hợp lệ."""
        if not filename:
            return ""
        sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:200] # Giới hạn độ dài tên file

    def start_requests(self):
        spider_logger.critical("<<<<< TVPL_SPIDER: START_REQUESTS CALLED >>>>>")
        for url in self.start_urls:
            spider_logger.info(f"Yielding initial Playwright request to: {url}")
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "p.nqTitle > a", timeout=60000),
                    ]
                },
                callback=self.parse_search_results,
                errback=self.errback_close_page # Thêm errback để đóng page khi có lỗi
            )

    async def parse_search_results(self, response):
        page = response.meta.get("playwright_page")
        try:
            self.search_pages_crawled_count += 1
            self.logger.info(f"Parsing search results page: {response.url} (Crawl count: {self.search_pages_crawled_count})")

            document_links = response.css('p.nqTitle > a::attr(href)').getall()
            self.logger.info(f"Found {len(document_links)} document links on page.")

            for doc_link in document_links:
                full_doc_url = response.urljoin(doc_link)
                if "/van-ban/" in full_doc_url:
                    ## FIX 2: Tối ưu hóa request đến trang chi tiết bằng cách chờ selector cụ thể
                    ## thay vì chờ sự kiện 'load' mặc định của trang.
                    yield scrapy.Request(
                        full_doc_url,
                        meta={
                            "playwright": True,
                            "playwright_include_page": True, # Yêu cầu page object để dùng
                            "playwright_page_methods": [
                                # Chờ cho đến khi một trong các selector chứa nội dung chính xuất hiện.
                                # Đây là cách hiệu quả hơn nhiều so với việc chờ toàn bộ trang tải xong.
                                PageMethod("wait_for_selector",
                                           "#divContentDoc, .content-htimcontent, #contentDoc, .content1",
                                           timeout=60000), # Timeout riêng cho hành động này

                                # Thử click vào lược đồ nếu có, không báo lỗi nếu không có
                                PageMethod("evaluate", "document.getElementById('ctl00_Content_ThongTinVB_spLuocDo')?.click()"),
                                PageMethod("wait_for_timeout", 2000), # Chờ một chút để JS có thể chạy (nếu có)
                            ],
                        },
                        callback=self.parse_document_detail,
                        errback=self.errback_close_page
                    )

            # Logic phân trang
            if self.MAX_SEARCH_RESULT_PAGES == 0 or self.search_pages_crawled_count < self.MAX_SEARCH_RESULT_PAGES:
                next_page_href = response.xpath("//div[@class='cmPager']/a[normalize-space(text())='Trang sau']/@href").get()
                if next_page_href:
                    next_page_url = response.urljoin(next_page_href)
                    self.logger.info(f"Following to next page: {next_page_url}")
                    yield scrapy.Request(
                        next_page_url,
                        meta={
                            "playwright": True,
                            "playwright_include_page": True,
                            "playwright_page_methods": [
                                PageMethod("wait_for_selector", "p.nqTitle > a", timeout=60000),
                            ]
                        },
                        callback=self.parse_search_results,
                        errback=self.errback_close_page
                    )
                else:
                    self.logger.info("No 'Trang sau' link found. Pagination likely ended.")
            else:
                self.logger.info(f"Reached MAX_SEARCH_RESULT_PAGES ({self.MAX_SEARCH_RESULT_PAGES}). Stopping pagination.")
        except Exception as e:
            self.logger.error(f"Error in parse_search_results for {response.url}: {e}", exc_info=True)
        finally:
            if page and not page.is_closed():
                await page.close()

    async def parse_document_detail(self, response):
        self.logger.info(f"Parsing document detail for: {response.url}")
        page = response.meta.get("playwright_page")

        try:
            # BeautifulSoup vẫn có thể được dùng với nội dung đã tải qua Playwright
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Trích xuất tiêu đề để làm tên file
            title_text = None
            title_selectors = ["div.vbProperties > .Title", "h1.page-title", "div.breadcrumbs + h1"]
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title_text = title_element.get_text(strip=True)
                    if title_text:
                        break
            
            if not title_text:
                html_title_tag = soup.find('title')
                if html_title_tag:
                    title_text = html_title_tag.get_text(strip=True).replace(" - THƯ VIỆN PHÁP LUẬT", "").strip()

            # Trích xuất nội dung chính
            raw_text_content = self.extract_raw_text(soup, response.url)

            # Kiểm tra nếu có nội dung để lưu
            if not raw_text_content or "Nội dung không thể trích xuất tự động" in raw_text_content:
                self.logger.warning(f"Content not extractable or empty for {response.url}. Skipping file save.")
                return

            # Tạo tên file
            if title_text and len(title_text) > 5:
                base_filename = self.sanitize_filename(title_text)
            else:
                self.logger.warning(f"Could not find a good title for {response.url}. Using URL hash as filename.")
                base_filename = hashlib.md5(response.url.encode()).hexdigest()

            filename = f"{base_filename}.txt"
            file_path = os.path.join(self.output_dir, filename)

            # Lưu nội dung vào file txt
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(raw_text_content)
                self.logger.info(f"Successfully saved content to: {file_path}")
            except IOError as e:
                self.logger.error(f"Could not write file {file_path}. Error: {e}")

        except Exception as e:
            self.logger.error(f"Critical error parsing document detail {response.url}: {e}", exc_info=True)
        finally:
            if page and not page.is_closed():
                await page.close()

    def extract_raw_text(self, soup, doc_url="Unknown URL"):
        """Trích xuất nội dung văn bản sạch từ soup HTML."""
        content_element = soup.select_one('#divContentDoc, .content-htimcontent, #contentDoc, .content1')
        
        if not content_element:
            self.logger.warning(f"Primary content selector not found for {doc_url}. Content may be incomplete.")
            content_element = soup.find('body')
            if not content_element:
                 return "Nội dung không thể trích xuất tự động."

        tmp_soup = BeautifulSoup(str(content_element), 'html.parser')

        elements_to_remove = [
            'script', 'style', 'header', 'footer', 'nav', 'form', 'iframe',
            '.vbInfoBox', '.vbInfo', '.att', '.boxdd', '.fastView', '.VBLQ', '.box-tag',
            '.download', '.social-buttons', '.toolbar', '.ads', '.related-post', '.breadcrumbs',
            '.page_toolbar', '.tools', '#comments', '.comment-section', '.sidebar', 'h1.page-title',
            '.doc-title', '.toc', '.luocdo', '#ctl00_Content_ctlDocInteraction', '.TitleVBContent',
            'div[id*="Ad"], div[class*="ad-"], .adsbygoogle'
        ]
        
        for selector in elements_to_remove:
            for tag in tmp_soup.select(selector):
                tag.decompose()
        
        text = tmp_soup.get_text(separator='\n', strip=True)
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)
        
        if len(text) < 150:
            self.logger.warning(f"Extracted text is very short ({len(text)} chars) for {doc_url}.")
            # Giữ lại nội dung ngắn thay vì loại bỏ hoàn toàn, có thể là văn bản ngắn thực sự
            # return "Nội dung không thể trích xuất tự động."

        return text.strip()
    
    ## FIX 3: Thêm một hàm callback lỗi (errback) để đảm bảo page được đóng
    ## ngay cả khi request gặp lỗi (ví dụ: TimeoutError).
    async def errback_close_page(self, failure):
        page = failure.request.meta.get("playwright_page")
        self.logger.error(f"Request failed for {failure.request.url}. Error: {failure.value}")
        if page and not page.is_closed():
            self.logger.warning(f"Closing page due to an error for request: {failure.request.url}")
            await page.close()