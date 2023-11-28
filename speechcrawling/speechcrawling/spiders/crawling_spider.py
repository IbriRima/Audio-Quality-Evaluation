from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class CrawlSpiderExample(CrawlSpider):
    name = "sp_crawler"
    allowed_domains = ["freesound.org"]
    start_urls = ["https://freesound.org/"]

    rules = (
        Rule(LinkExtractor(allow="people/"), callback='parse_item'),
    )

    def parse_item(self, response):

        container = response.css("div.container") 
        
        if container:
            bw_players = container.css('.bw-player')

            for bw_player in bw_players:
                mp3_url = bw_player.attrib.get('data-mp3')
                if mp3_url:
                    self.logger.info(f'MP3 URL: {mp3_url}')
                    yield {
                        "original_url": response.url,
                        "record":mp3_url   
                    }
        
                    
                else:
                    self.logger.warning('data-mp3 attribute not found within the .bw-player element.')
        else:
            self.logger.warning('Container element not found.')
